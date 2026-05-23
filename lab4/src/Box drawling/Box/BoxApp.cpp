#include "../../Common/d3dApp.h"
#include "../../Common/MathHelper.h"
#include "../../Common/UploadBuffer.h"
#include "../../Common/d3dUtil.h"
#include "../../Common/FreeCamera.h"
#include "../../Common/GeometryGenerator.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "GBuffer.h"
#include "RenderingSystem.h"

#include <vector>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <array>
#include <cmath>
#include <limits>
#include <random>

using Microsoft::WRL::ComPtr;
using namespace DirectX;

namespace
{
    constexpr UINT kSceneSponza = 0;
    constexpr UINT kSceneForest = 1;
    constexpr UINT kInvalidIndex = (std::numeric_limits<UINT>::max)();
    constexpr float kTreeBillboardDistance = 30.0f;
    constexpr float kTreeBillboardDistanceSq = kTreeBillboardDistance * kTreeBillboardDistance;
}

struct Vertex
{
    XMFLOAT3 Pos;
    XMFLOAT3 Normal;
    XMFLOAT3 Tangent;
    XMFLOAT2 TexC;
};

struct ObjectConstants
{
    XMFLOAT4X4 World = MathHelper::Identity4x4();
    XMFLOAT4X4 ViewProj = MathHelper::Identity4x4();
    XMFLOAT2 UvScale = XMFLOAT2(1.0f, 1.0f);
    XMFLOAT2 UvOffset = XMFLOAT2(0.0f, 0.0f);
    XMFLOAT4 Tint = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
    XMFLOAT3 CameraPosW = XMFLOAT3(0.0f, 0.0f, 0.0f);
    float DisplacementScale = 0.0f;
    float TessMinDistance = 3.0f;
    float TessMaxDistance = 18.0f;
    float TessMinFactor = 1.0f;
    float TessMaxFactor = 12.0f;
};

struct TgaImage
{
    int Width = 0;
    int Height = 0;
    std::vector<uint8_t> Rgba;
};

static TgaImage LoadTgaRgba(const std::wstring& filePath)
{
    std::ifstream in(filePath, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to open .tga file");

    uint8_t header[18] = {};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in)
        throw std::runtime_error("Failed to read .tga header");

    const uint8_t idLength = header[0];
    const uint8_t imageType = header[2];
    const uint16_t width = static_cast<uint16_t>(header[12] | (header[13] << 8));
    const uint16_t height = static_cast<uint16_t>(header[14] | (header[15] << 8));
    const uint8_t bpp = header[16];
    const uint8_t imageDesc = header[17];

    if (imageType != 2 || (bpp != 24 && bpp != 32))
        throw std::runtime_error("Only uncompressed 24/32-bit TGA is supported");

    if (idLength > 0)
        in.seekg(idLength, std::ios::cur);

    const int pixelBytes = bpp / 8;
    const size_t srcSize = static_cast<size_t>(width) * height * pixelBytes;
    std::vector<uint8_t> src(srcSize);
    in.read(reinterpret_cast<char*>(src.data()), srcSize);
    if (!in)
        throw std::runtime_error("Failed to read .tga pixel data");

    TgaImage img;
    img.Width = width;
    img.Height = height;
    img.Rgba.resize(static_cast<size_t>(width) * height * 4);

    const bool topOrigin = (imageDesc & 0x20) != 0;

    for (int y = 0; y < height; ++y)
    {
        const int srcY = topOrigin ? y : (height - 1 - y);
        for (int x = 0; x < width; ++x)
        {
            const size_t s = (static_cast<size_t>(srcY) * width + x) * pixelBytes;
            const size_t d = (static_cast<size_t>(y) * width + x) * 4;
            img.Rgba[d + 0] = src[s + 2];
            img.Rgba[d + 1] = src[s + 1];
            img.Rgba[d + 2] = src[s + 0];
            img.Rgba[d + 3] = (pixelBytes == 4) ? src[s + 3] : 255;
        }
    }

    return img;
}

namespace
{
    BoundingBox MakeBoundingBox(const XMFLOAT3& minPoint, const XMFLOAT3& maxPoint)
    {
        const XMFLOAT3 center(
            0.5f * (minPoint.x + maxPoint.x),
            0.5f * (minPoint.y + maxPoint.y),
            0.5f * (minPoint.z + maxPoint.z));

        const XMFLOAT3 extents(
            0.5f * (maxPoint.x - minPoint.x),
            0.5f * (maxPoint.y - minPoint.y),
            0.5f * (maxPoint.z - minPoint.z));

        return BoundingBox(center, extents);
    }

    BoundingBox ComputeLocalBounds(const std::vector<Vertex>& vertices)
    {
        XMFLOAT3 minPoint(
            (std::numeric_limits<float>::max)(),
            (std::numeric_limits<float>::max)(),
            (std::numeric_limits<float>::max)());
        XMFLOAT3 maxPoint(
            -(std::numeric_limits<float>::max)(),
            -(std::numeric_limits<float>::max)(),
            -(std::numeric_limits<float>::max)());

        for (const Vertex& vertex : vertices)
        {
            minPoint.x = (std::min)(minPoint.x, vertex.Pos.x);
            minPoint.y = (std::min)(minPoint.y, vertex.Pos.y);
            minPoint.z = (std::min)(minPoint.z, vertex.Pos.z);

            maxPoint.x = (std::max)(maxPoint.x, vertex.Pos.x);
            maxPoint.y = (std::max)(maxPoint.y, vertex.Pos.y);
            maxPoint.z = (std::max)(maxPoint.z, vertex.Pos.z);
        }

        return MakeBoundingBox(minPoint, maxPoint);
    }

    BoundingBox TransformBounds(const BoundingBox& bounds, const XMFLOAT4X4& world)
    {
        BoundingBox transformed;
        bounds.Transform(transformed, XMLoadFloat4x4(&world));
        return transformed;
    }

    BoundingBox MergeBounds(const BoundingBox& a, const BoundingBox& b)
    {
        const XMFLOAT3 minPoint(
            (std::min)(a.Center.x - a.Extents.x, b.Center.x - b.Extents.x),
            (std::min)(a.Center.y - a.Extents.y, b.Center.y - b.Extents.y),
            (std::min)(a.Center.z - a.Extents.z, b.Center.z - b.Extents.z));

        const XMFLOAT3 maxPoint(
            (std::max)(a.Center.x + a.Extents.x, b.Center.x + b.Extents.x),
            (std::max)(a.Center.y + a.Extents.y, b.Center.y + b.Extents.y),
            (std::max)(a.Center.z + a.Extents.z, b.Center.z + b.Extents.z));

        return MakeBoundingBox(minPoint, maxPoint);
    }

    XMFLOAT3 BuildFallbackTangent(const XMFLOAT3& normal)
    {
        XMVECTOR n = XMVector3Normalize(XMLoadFloat3(&normal));
        XMVECTOR ref = (std::fabs(normal.y) > 0.99f) ? XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f) : XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
        XMVECTOR tangent = XMVector3Cross(ref, n);
        if (XMVectorGetX(XMVector3LengthSq(tangent)) < 1e-8f)
            tangent = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);

        XMFLOAT3 result;
        XMStoreFloat3(&result, XMVector3Normalize(tangent));
        return result;
    }

    void ComputeTangents(std::vector<Vertex>& vertices, const std::vector<std::uint32_t>& indices)
    {
        std::vector<XMFLOAT3> accumulated(vertices.size(), XMFLOAT3(0.0f, 0.0f, 0.0f));

        for (size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            const uint32_t i0 = indices[i + 0];
            const uint32_t i1 = indices[i + 1];
            const uint32_t i2 = indices[i + 2];

            const Vertex& v0 = vertices[i0];
            const Vertex& v1 = vertices[i1];
            const Vertex& v2 = vertices[i2];

            const XMVECTOR p0 = XMLoadFloat3(&v0.Pos);
            const XMVECTOR p1 = XMLoadFloat3(&v1.Pos);
            const XMVECTOR p2 = XMLoadFloat3(&v2.Pos);

            const XMVECTOR e1 = p1 - p0;
            const XMVECTOR e2 = p2 - p0;

            const float du1 = v1.TexC.x - v0.TexC.x;
            const float dv1 = v1.TexC.y - v0.TexC.y;
            const float du2 = v2.TexC.x - v0.TexC.x;
            const float dv2 = v2.TexC.y - v0.TexC.y;

            XMVECTOR tangent = XMVectorZero();
            const float denom = du1 * dv2 - dv1 * du2;
            if (std::fabs(denom) > 1e-6f)
            {
                tangent = (e1 * dv2 - e2 * dv1) / denom;
            }
            else
            {
                tangent = XMLoadFloat3(&BuildFallbackTangent(v0.Normal));
            }

            XMFLOAT3 tangentF;
            XMStoreFloat3(&tangentF, tangent);

            accumulated[i0].x += tangentF.x;
            accumulated[i0].y += tangentF.y;
            accumulated[i0].z += tangentF.z;
            accumulated[i1].x += tangentF.x;
            accumulated[i1].y += tangentF.y;
            accumulated[i1].z += tangentF.z;
            accumulated[i2].x += tangentF.x;
            accumulated[i2].y += tangentF.y;
            accumulated[i2].z += tangentF.z;
        }

        for (size_t i = 0; i < vertices.size(); ++i)
        {
            const XMVECTOR normal = XMVector3Normalize(XMLoadFloat3(&vertices[i].Normal));
            XMVECTOR tangent = XMLoadFloat3(&accumulated[i]);
            tangent = tangent - normal * XMVector3Dot(normal, tangent);

            if (XMVectorGetX(XMVector3LengthSq(tangent)) < 1e-8f)
                tangent = XMLoadFloat3(&BuildFallbackTangent(vertices[i].Normal));

            XMStoreFloat3(&vertices[i].Tangent, XMVector3Normalize(tangent));
        }
    }
}

class DeferredRenderer
{
public:
    GBuffer Buffers;
    RenderingSystem Lighting;
};

class BoxApp : public D3DApp
{
public:
    BoxApp(HINSTANCE hInstance);
    BoxApp(const BoxApp& rhs) = delete;
    BoxApp& operator=(const BoxApp& rhs) = delete;
    ~BoxApp();

    virtual bool Initialize()override;

private:
    virtual void OnResize()override;
    virtual void Update(const GameTimer& gt)override;
    virtual void Draw(const GameTimer& gt)override;
    virtual std::wstring GetAdditionalWindowText() const override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y)override;
    virtual void OnMouseUp(WPARAM btnState, int x, int y)override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y)override;

    struct CullObject;
    struct OctreeNode;

    void SetActiveScene(UINT sceneId);
    void UpdateVisibleBatches();
    UINT SelectTreeBatch(const CullObject& object, const XMFLOAT3& cameraPos) const;
    void AddVisibleTreeObject(UINT objectIndex, const XMFLOAT3& cameraPos, std::vector<UINT>& visibleBatches,
        UINT& visibleObjects, UINT& visibleMeshObjects, UINT& visibleBillboardObjects) const;
    void BuildForestOctree();
    void InsertIntoOctree(OctreeNode& node, UINT objectIndex, int depth);
    void QueryOctree(const OctreeNode& node, const BoundingFrustum& frustum, const XMFLOAT3& cameraPos,
        std::vector<UINT>& visibleBatches, UINT& visibleObjects, UINT& visibleMeshObjects, UINT& visibleBillboardObjects) const;
    void CollectOctreeBatches(const OctreeNode& node, const XMFLOAT3& cameraPos,
        std::vector<UINT>& visibleBatches, UINT& visibleObjects, UINT& visibleMeshObjects, UINT& visibleBillboardObjects) const;
    void BuildDescriptorHeaps();
    void BuildDeferredSrvHeap();
    void UpdateDeferredSrvDescriptors();
    void BuildConstantBuffers();
    void BuildRootSignatures();
    void BuildShadersAndInputLayout();
    void BuildBoxGeometry();
    void BuildPSOs();
    void BuildLights();
    UINT CreateRgbaTexture(const std::string& name, UINT width, UINT height, const std::vector<std::uint8_t>& rgba);
    UINT CreateSolidColorTexture(const std::string& name, const std::array<std::uint8_t, 4>& rgba);
    UINT LoadOrCreateTexture(const std::filesystem::path& baseDir, const std::string& texName);

private:
    struct DrawBatch
    {
        UINT IndexCount = 0;
        UINT StartIndexLocation = 0;
        UINT DiffuseSrvIndex = 0;
        UINT NormalSrvIndex = 0;
        UINT DisplacementSrvIndex = 0;
        UINT SceneId = kSceneSponza;
        UINT CullObjectIndex = kInvalidIndex;
        bool Tessellated = false;
        bool AnimateUv = false;
        bool Cullable = false;
        bool Billboard = false;
        XMFLOAT2 UvScale = XMFLOAT2(1.0f, 1.0f);
        float DisplacementScale = 0.0f;
        float BillboardWidth = 1.0f;
        float BillboardHeight = 1.0f;
        XMFLOAT3 BillboardBase = XMFLOAT3(0.0f, 0.0f, 0.0f);
        float Padding = 0.0f;
        XMFLOAT4 Tint = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
        XMFLOAT4X4 World = MathHelper::Identity4x4();
        BoundingBox Bounds;
    };

    struct CullObject
    {
        BoundingBox Bounds;
        UINT MeshBatchIndex = 0;
        UINT BillboardBatchIndex = 0;
        XMFLOAT3 Position = XMFLOAT3(0.0f, 0.0f, 0.0f);
        float BillboardWidth = 1.0f;
        float BillboardHeight = 1.0f;
    };

    struct OctreeNode
    {
        BoundingBox Bounds;
        BoundingBox LooseBounds;
        std::vector<UINT> ObjectIndices;
        std::array<std::unique_ptr<OctreeNode>, 8> Children;
    };

    ComPtr<ID3D12RootSignature> mGeometryRootSignature = nullptr;
    ComPtr<ID3D12RootSignature> mLightingRootSignature = nullptr;
    ComPtr<ID3D12DescriptorHeap> mCbvSrvHeap = nullptr;
    ComPtr<ID3D12DescriptorHeap> mDeferredSrvHeap = nullptr;

    std::unique_ptr<UploadBuffer<ObjectConstants>> mObjectCB = nullptr;
    std::unique_ptr<UploadBuffer<DeferredPassConstants>> mDeferredCB = nullptr;
    DeferredRenderer mDeferredRenderer;

    std::unique_ptr<MeshGeometry> mBoxGeo = nullptr;
    std::vector<DrawBatch> mDrawBatches;
    std::vector<UINT> mVisibleBatchIndices;
    std::vector<CullObject> mForestObjects;
    std::unique_ptr<OctreeNode> mForestOctreeRoot = nullptr;
    std::vector<std::unique_ptr<Texture>> mTextures;
    std::unordered_map<std::string, UINT> mTextureIndexByName;

    ComPtr<ID3DBlob> mGBufferVS = nullptr;
    ComPtr<ID3DBlob> mGBufferPS = nullptr;
    ComPtr<ID3DBlob> mGBufferTessVS = nullptr;
    ComPtr<ID3DBlob> mGBufferHS = nullptr;
    ComPtr<ID3DBlob> mGBufferDS = nullptr;
    ComPtr<ID3DBlob> mLightingVS = nullptr;
    ComPtr<ID3DBlob> mLightingPS = nullptr;
    ComPtr<ID3DBlob> mDebugWirePS = nullptr;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

    ComPtr<ID3D12PipelineState> mGBufferPSO = nullptr;
    ComPtr<ID3D12PipelineState> mGBufferBillboardPSO = nullptr;
    ComPtr<ID3D12PipelineState> mGBufferTessPSO = nullptr;
    ComPtr<ID3D12PipelineState> mLightingPSO = nullptr;
    ComPtr<ID3D12PipelineState> mTessWirePSO = nullptr;

    FreeCamera mCamera;
    float mMoveSpeed = 10.0f;
    float mLookSpeed = 2.0f;

    UINT mDebugView = 0;
    bool mDebugViewKeyWasDown = false;
    bool mTextureAnimationEnabled = true;
    bool mTextureAnimationKeyWasDown = false;
    float mUvAnimSpeedU = 0.035f;
    float mUvAnimSpeedV = 0.018f;
    XMFLOAT2 mAnimatedUvOffset = XMFLOAT2(0.0f, 0.0f);
    float mTessMinDistance = 3.0f;
    float mTessMaxDistance = 20.0f;
    float mTessMinFactor = 1.0f;
    float mTessMaxFactor = 12.0f;
    UINT mActiveScene = kSceneSponza;
    UINT mVisibleCullObjectCount = 0;
    UINT mVisibleTreeMeshCount = 0;
    UINT mVisibleTreeBillboardCount = 0;
    bool mScene1KeyWasDown = false;
    bool mScene2KeyWasDown = false;
    bool mFrustumCullingEnabled = true;
    bool mFrustumCullingKeyWasDown = false;
    bool mOctreeCullingEnabled = true;
    bool mOctreeCullingKeyWasDown = false;

    POINT mLastMousePos;
};

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try
    {
        BoxApp theApp(hInstance);
        if (!theApp.Initialize())
            return 0;

        return theApp.Run();
    }
    catch (DxException& e)
    {
        MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
        return 0;
    }
}

BoxApp::BoxApp(HINSTANCE hInstance)
    : D3DApp(hInstance)
{
    mCamera.SetPosition(0.0f, 2.2f, -8.5f);
}

BoxApp::~BoxApp()
{
}

bool BoxApp::Initialize()
{
    if (!D3DApp::Initialize())
        return false;

    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    BuildShadersAndInputLayout();
    BuildBoxGeometry();
    BuildConstantBuffers();
    BuildDescriptorHeaps();
    BuildDeferredSrvHeap();
    BuildRootSignatures();
    BuildPSOs();
    mDeferredRenderer.Buffers.Build(md3dDevice.Get(), mClientWidth, mClientHeight);
    UpdateDeferredSrvDescriptors();
    BuildLights();
    mCamera.UpdateViewMatrix();
    UpdateVisibleBatches();

    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);
    FlushCommandQueue();

    return true;
}

void BoxApp::OnResize()
{
    D3DApp::OnResize();

    mCamera.SetLens(0.25f * MathHelper::Pi, AspectRatio(), 1.0f, 1000.0f);
    if (mDeferredRenderer.Buffers.IsInitialized())
    {
        mDeferredRenderer.Buffers.Resize(md3dDevice.Get(), mClientWidth, mClientHeight);
        UpdateDeferredSrvDescriptors();
    }
}

void BoxApp::SetActiveScene(UINT sceneId)
{
    if (mActiveScene == sceneId)
        return;

    mActiveScene = sceneId;

    if (mActiveScene == kSceneForest)
    {
        mMoveSpeed = 32.0f;
        mCamera.SetPosition(0.0f, 5.0f, -72.0f);
    }
    else
    {
        mMoveSpeed = 10.0f;
        mCamera.SetPosition(0.0f, 2.2f, -8.5f);
    }
}

void BoxApp::UpdateVisibleBatches()
{
    mVisibleBatchIndices.clear();
    mVisibleCullObjectCount = 0;
    mVisibleTreeMeshCount = 0;
    mVisibleTreeBillboardCount = 0;

    auto addSceneNonCullableBatches = [&]()
    {
        for (UINT i = 0; i < static_cast<UINT>(mDrawBatches.size()); ++i)
        {
            const DrawBatch& batch = mDrawBatches[i];
            if (batch.SceneId == mActiveScene && !batch.Cullable)
                mVisibleBatchIndices.push_back(i);
        }
    };

    addSceneNonCullableBatches();

    if (mActiveScene != kSceneForest)
        return;

    const XMFLOAT3 cameraPos = mCamera.GetPosition3f();

    if (!mFrustumCullingEnabled)
    {
        for (UINT objectIndex = 0; objectIndex < static_cast<UINT>(mForestObjects.size()); ++objectIndex)
        {
            AddVisibleTreeObject(objectIndex, cameraPos, mVisibleBatchIndices, mVisibleCullObjectCount,
                mVisibleTreeMeshCount, mVisibleTreeBillboardCount);
        }
        return;
    }

    BoundingFrustum viewSpaceFrustum;
    BoundingFrustum::CreateFromMatrix(viewSpaceFrustum, mCamera.GetProj());

    BoundingFrustum worldSpaceFrustum;
    const XMMATRIX invView = XMMatrixInverse(nullptr, mCamera.GetView());
    viewSpaceFrustum.Transform(worldSpaceFrustum, invView);

    if (mOctreeCullingEnabled && mForestOctreeRoot)
    {
        QueryOctree(*mForestOctreeRoot, worldSpaceFrustum, cameraPos, mVisibleBatchIndices, mVisibleCullObjectCount,
            mVisibleTreeMeshCount, mVisibleTreeBillboardCount);
    }
    else
    {
        for (UINT objectIndex = 0; objectIndex < static_cast<UINT>(mForestObjects.size()); ++objectIndex)
        {
            const CullObject& object = mForestObjects[objectIndex];
            if (worldSpaceFrustum.Contains(object.Bounds) != DISJOINT)
            {
                AddVisibleTreeObject(objectIndex, cameraPos, mVisibleBatchIndices, mVisibleCullObjectCount,
                    mVisibleTreeMeshCount, mVisibleTreeBillboardCount);
            }
        }
    }
}

UINT BoxApp::SelectTreeBatch(const CullObject& object, const XMFLOAT3& cameraPos) const
{
    const XMFLOAT3& center = object.Bounds.Center;
    const float dx = cameraPos.x - center.x;
    const float dy = cameraPos.y - center.y;
    const float dz = cameraPos.z - center.z;
    const float distSq = dx * dx + dy * dy + dz * dz;
    return (distSq > kTreeBillboardDistanceSq) ? object.BillboardBatchIndex : object.MeshBatchIndex;
}

void BoxApp::AddVisibleTreeObject(UINT objectIndex, const XMFLOAT3& cameraPos, std::vector<UINT>& visibleBatches,
    UINT& visibleObjects, UINT& visibleMeshObjects, UINT& visibleBillboardObjects) const
{
    const CullObject& object = mForestObjects[objectIndex];
    const UINT batchIndex = SelectTreeBatch(object, cameraPos);
    visibleBatches.push_back(batchIndex);
    ++visibleObjects;

    if (batchIndex == object.BillboardBatchIndex)
        ++visibleBillboardObjects;
    else
        ++visibleMeshObjects;
}

std::wstring BoxApp::GetAdditionalWindowText() const
{
    const XMFLOAT3 cameraPos = mCamera.GetPosition3f();
    static const std::array<const wchar_t*, 5> kDebugViewNames =
    {
        L"Lit",
        L"Albedo",
        L"Normal",
        L"Depth",
        L"Tess Wire"
    };
    static const std::array<const wchar_t*, 2> kSceneNames =
    {
        L"Sponza",
        L"Forest"
    };

    const wchar_t* cullingMode = L"off";
    if (mFrustumCullingEnabled)
        cullingMode = mOctreeCullingEnabled ? L"frustum+octree" : L"frustum";

    std::wostringstream stream;
    stream << std::fixed << std::setprecision(2)
        << L"cam xyz: (" << cameraPos.x << L", " << cameraPos.y << L", " << cameraPos.z << L")"
        << L" | scene: " << kSceneNames[mActiveScene]
        << L" | debug: " << kDebugViewNames[mDebugView]
        << L" | tex anim: " << (mTextureAnimationEnabled ? L"on" : L"off")
        << L" | culling: " << cullingMode
        << L" | visible: " << mVisibleCullObjectCount << L"/" << mForestObjects.size()
        << L" | lod mesh/bb: " << mVisibleTreeMeshCount << L"/" << mVisibleTreeBillboardCount;

    return stream.str();
}

void BoxApp::Update(const GameTimer& gt)
{
    const float dt = gt.DeltaTime();
    const float moveStep = mMoveSpeed * dt;

    if (d3dUtil::IsKeyDown('W'))
        mCamera.Walk(moveStep);
    if (d3dUtil::IsKeyDown('S'))
        mCamera.Walk(-moveStep);
    if (d3dUtil::IsKeyDown('A'))
        mCamera.Strafe(-moveStep);
    if (d3dUtil::IsKeyDown('D'))
        mCamera.Strafe(moveStep);
    if (d3dUtil::IsKeyDown('Q'))
        mCamera.Rise(-moveStep);
    if (d3dUtil::IsKeyDown('E'))
        mCamera.Rise(moveStep);

    const bool scene1KeyDown = d3dUtil::IsKeyDown('1');
    if (scene1KeyDown && !mScene1KeyWasDown)
        SetActiveScene(kSceneSponza);
    mScene1KeyWasDown = scene1KeyDown;

    const bool scene2KeyDown = d3dUtil::IsKeyDown('2');
    if (scene2KeyDown && !mScene2KeyWasDown)
        SetActiveScene(kSceneForest);
    mScene2KeyWasDown = scene2KeyDown;

    const bool debugKeyDown = d3dUtil::IsKeyDown('G');
    if (debugKeyDown && !mDebugViewKeyWasDown)
        mDebugView = (mDebugView + 1) % 5;
    mDebugViewKeyWasDown = debugKeyDown;

    const bool textureAnimationKeyDown = d3dUtil::IsKeyDown('T');
    if (textureAnimationKeyDown && !mTextureAnimationKeyWasDown)
        mTextureAnimationEnabled = !mTextureAnimationEnabled;
    mTextureAnimationKeyWasDown = textureAnimationKeyDown;

    const bool frustumCullingKeyDown = d3dUtil::IsKeyDown('F');
    if (frustumCullingKeyDown && !mFrustumCullingKeyWasDown)
        mFrustumCullingEnabled = !mFrustumCullingEnabled;
    mFrustumCullingKeyWasDown = frustumCullingKeyDown;

    const bool octreeCullingKeyDown = d3dUtil::IsKeyDown('O');
    if (octreeCullingKeyDown && !mOctreeCullingKeyWasDown)
        mOctreeCullingEnabled = !mOctreeCullingEnabled;
    mOctreeCullingKeyWasDown = octreeCullingKeyDown;

    if (mTextureAnimationEnabled)
    {
        mAnimatedUvOffset.x += dt * mUvAnimSpeedU;
        mAnimatedUvOffset.y += dt * mUvAnimSpeedV;
    }

    mCamera.UpdateViewMatrix();
    UpdateVisibleBatches();

    const XMMATRIX view = mCamera.GetView();
    const XMMATRIX proj = mCamera.GetProj();
    const XMMATRIX viewProj = view * proj;
    const XMFLOAT3 cameraPos = mCamera.GetPosition3f();

    for (UINT batchIndex : mVisibleBatchIndices)
    {
        DrawBatch& batch = mDrawBatches[batchIndex];
        if (!batch.Billboard)
            continue;

        const float dx = cameraPos.x - batch.BillboardBase.x;
        const float dz = cameraPos.z - batch.BillboardBase.z;
        if (dx * dx + dz * dz < 1e-6f)
            continue;

        const float yaw = std::atan2(dx, dz);
        XMStoreFloat4x4(&batch.World,
            XMMatrixScaling(batch.BillboardWidth, batch.BillboardHeight, 1.0f) *
            XMMatrixRotationY(yaw) *
            XMMatrixTranslation(batch.BillboardBase.x, batch.BillboardBase.y, batch.BillboardBase.z));
    }

    for (size_t i = 0; i < mDrawBatches.size(); ++i)
    {
        const DrawBatch& batch = mDrawBatches[i];

        ObjectConstants objConstants;
        const XMMATRIX world = XMLoadFloat4x4(&batch.World);
        XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));
        XMStoreFloat4x4(&objConstants.ViewProj, XMMatrixTranspose(viewProj));
        objConstants.UvScale = batch.UvScale;
        objConstants.UvOffset = batch.AnimateUv ? mAnimatedUvOffset : XMFLOAT2(0.0f, 0.0f);
        objConstants.Tint = batch.Tint;
        objConstants.CameraPosW = cameraPos;
        objConstants.DisplacementScale = batch.DisplacementScale;
        objConstants.TessMinDistance = mTessMinDistance;
        objConstants.TessMaxDistance = mTessMaxDistance;
        objConstants.TessMinFactor = mTessMinFactor;
        objConstants.TessMaxFactor = mTessMaxFactor;
        mObjectCB->CopyData(static_cast<int>(i), objConstants);
    }

    DeferredPassConstants pass = {};
    const XMMATRIX invView = XMMatrixInverse(nullptr, view);
    const XMMATRIX invProj = XMMatrixInverse(nullptr, proj);
    XMStoreFloat4x4(&pass.InvView, XMMatrixTranspose(invView));
    XMStoreFloat4x4(&pass.InvProj, XMMatrixTranspose(invProj));
    pass.CameraPosW = cameraPos;
    pass.PointLightCount = static_cast<UINT>(std::min<size_t>(mDeferredRenderer.Lighting.PointLights.size(), 16));
    pass.DirectionalLightCount = static_cast<UINT>(std::min<size_t>(mDeferredRenderer.Lighting.DirectionalLights.size(), 8));
    pass.SpotLightCount = static_cast<UINT>(std::min<size_t>(mDeferredRenderer.Lighting.SpotLights.size(), 8));
    pass.DebugView = mDebugView;
    pass.Ambient = (mActiveScene == kSceneForest) ? 0.12f : 0.001f;

    for (UINT i = 0; i < pass.PointLightCount; ++i)
        pass.PointLights[i] = mDeferredRenderer.Lighting.PointLights[i];
    for (UINT i = 0; i < pass.DirectionalLightCount; ++i)
        pass.DirectionalLights[i] = mDeferredRenderer.Lighting.DirectionalLights[i];
    for (UINT i = 0; i < pass.SpotLightCount; ++i)
        pass.SpotLights[i] = mDeferredRenderer.Lighting.SpotLights[i];

    mDeferredCB->CopyData(0, pass);
}

void BoxApp::Draw(const GameTimer& gt)
{
    ThrowIfFailed(mDirectCmdListAlloc->Reset());

    auto* albedo = mDeferredRenderer.Buffers.AlbedoResource();
    auto* normal = mDeferredRenderer.Buffers.NormalResource();
    auto* depth = mDepthStencilBuffer.Get();

    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), mGBufferPSO.Get()));
    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    CD3DX12_RESOURCE_BARRIER preGeom[2] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition(albedo, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET),
        CD3DX12_RESOURCE_BARRIER::Transition(normal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)
    };
    mCommandList->ResourceBarrier(2, preGeom);

    const float normalClear[4] = { 0.5f, 0.5f, 0.0f, 1.0f };
    mCommandList->ClearRenderTargetView(mDeferredRenderer.Buffers.AlbedoRtv(), Colors::Black, 0, nullptr);
    mCommandList->ClearRenderTargetView(mDeferredRenderer.Buffers.NormalRtv(), normalClear, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    D3D12_CPU_DESCRIPTOR_HANDLE gbuffers[2] = { mDeferredRenderer.Buffers.AlbedoRtv(), mDeferredRenderer.Buffers.NormalRtv() };
    mCommandList->OMSetRenderTargets(2, gbuffers, false, &DepthStencilView());

    ID3D12DescriptorHeap* geomHeaps[] = { mCbvSrvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(geomHeaps), geomHeaps);
    mCommandList->SetGraphicsRootSignature(mGeometryRootSignature.Get());
    mCommandList->IASetVertexBuffers(0, 1, &mBoxGeo->VertexBufferView());
    mCommandList->IASetIndexBuffer(&mBoxGeo->IndexBufferView());

    const UINT descSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    const auto baseSrvHandle = mCbvSrvHeap->GetGPUDescriptorHandleForHeapStart();
    const D3D12_GPU_VIRTUAL_ADDRESS objectCbAddress = mObjectCB->Resource()->GetGPUVirtualAddress();

    auto drawBatches = [&](bool tessellated, bool billboard, ID3D12PipelineState* pso, D3D_PRIMITIVE_TOPOLOGY topology)
    {
        mCommandList->SetPipelineState(pso);
        mCommandList->IASetPrimitiveTopology(topology);

        for (UINT batchIndex : mVisibleBatchIndices)
        {
            const DrawBatch& batch = mDrawBatches[batchIndex];
            if (batch.Tessellated != tessellated || batch.Billboard != billboard)
                continue;

            auto diffuseHandle = baseSrvHandle;
            diffuseHandle.ptr += static_cast<SIZE_T>(batch.DiffuseSrvIndex) * descSize;
            auto normalHandle = baseSrvHandle;
            normalHandle.ptr += static_cast<SIZE_T>(batch.NormalSrvIndex) * descSize;
            auto displacementHandle = baseSrvHandle;
            displacementHandle.ptr += static_cast<SIZE_T>(batch.DisplacementSrvIndex) * descSize;

            mCommandList->SetGraphicsRootConstantBufferView(0, objectCbAddress + static_cast<UINT64>(batchIndex) * objCBByteSize);
            mCommandList->SetGraphicsRootDescriptorTable(1, diffuseHandle);
            mCommandList->SetGraphicsRootDescriptorTable(2, normalHandle);
            mCommandList->SetGraphicsRootDescriptorTable(3, displacementHandle);
            mCommandList->DrawIndexedInstanced(batch.IndexCount, 1, batch.StartIndexLocation, 0, 0);
        }
    };

    drawBatches(false, false, mGBufferPSO.Get(), D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    drawBatches(false, true, mGBufferBillboardPSO.Get(), D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    drawBatches(true, false, mGBufferTessPSO.Get(), D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);

    CD3DX12_RESOURCE_BARRIER toLighting[4] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition(albedo, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(normal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(depth, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE),
        CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET)
    };
    mCommandList->ResourceBarrier(4, toLighting);

    mCommandList->SetPipelineState(mLightingPSO.Get());
    mCommandList->SetGraphicsRootSignature(mLightingRootSignature.Get());
    D3D12_CPU_DESCRIPTOR_HANDLE backBufferRtv = CurrentBackBufferView();
    mCommandList->ClearRenderTargetView(backBufferRtv, Colors::Black, 0, nullptr);
    mCommandList->OMSetRenderTargets(1, &backBufferRtv, true, nullptr);

    ID3D12DescriptorHeap* lightHeaps[] = { mDeferredSrvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(lightHeaps), lightHeaps);
    mCommandList->SetGraphicsRootDescriptorTable(0, mDeferredSrvHeap->GetGPUDescriptorHandleForHeapStart());
    mCommandList->SetGraphicsRootConstantBufferView(1, mDeferredCB->Resource()->GetGPUVirtualAddress());
    mCommandList->IASetVertexBuffers(0, 0, nullptr);
    mCommandList->IASetIndexBuffer(nullptr);
    mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    mCommandList->DrawInstanced(3, 1, 0, 0);

    D3D12_RESOURCE_STATES depthStateBeforePresent = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    if (mDebugView == 4)
    {
        CD3DX12_RESOURCE_BARRIER debugDepthBarrier = CD3DX12_RESOURCE_BARRIER::Transition(
            depth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_READ);
        mCommandList->ResourceBarrier(1, &debugDepthBarrier);
        depthStateBeforePresent = D3D12_RESOURCE_STATE_DEPTH_READ;

        mCommandList->SetGraphicsRootSignature(mGeometryRootSignature.Get());
        mCommandList->SetDescriptorHeaps(_countof(geomHeaps), geomHeaps);
        mCommandList->SetPipelineState(mTessWirePSO.Get());
        mCommandList->OMSetRenderTargets(1, &backBufferRtv, false, &DepthStencilView());
        mCommandList->IASetVertexBuffers(0, 1, &mBoxGeo->VertexBufferView());
        mCommandList->IASetIndexBuffer(&mBoxGeo->IndexBufferView());
        mCommandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);

        for (UINT batchIndex : mVisibleBatchIndices)
        {
            const DrawBatch& batch = mDrawBatches[batchIndex];
            if (!batch.Tessellated)
                continue;

            auto diffuseHandle = baseSrvHandle;
            diffuseHandle.ptr += static_cast<SIZE_T>(batch.DiffuseSrvIndex) * descSize;
            auto normalHandle = baseSrvHandle;
            normalHandle.ptr += static_cast<SIZE_T>(batch.NormalSrvIndex) * descSize;
            auto displacementHandle = baseSrvHandle;
            displacementHandle.ptr += static_cast<SIZE_T>(batch.DisplacementSrvIndex) * descSize;

            mCommandList->SetGraphicsRootConstantBufferView(0, objectCbAddress + static_cast<UINT64>(batchIndex) * objCBByteSize);
            mCommandList->SetGraphicsRootDescriptorTable(1, diffuseHandle);
            mCommandList->SetGraphicsRootDescriptorTable(2, normalHandle);
            mCommandList->SetGraphicsRootDescriptorTable(3, displacementHandle);
            mCommandList->DrawIndexedInstanced(batch.IndexCount, 1, batch.StartIndexLocation, 0, 0);
        }
    }

    CD3DX12_RESOURCE_BARRIER endFrame[2] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition(depth, depthStateBeforePresent, D3D12_RESOURCE_STATE_DEPTH_WRITE),
        CD3DX12_RESOURCE_BARRIER::Transition(CurrentBackBuffer(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT)
    };
    mCommandList->ResourceBarrier(2, endFrame);

    ThrowIfFailed(mCommandList->Close());

    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    ThrowIfFailed(mSwapChain->Present(0, 0));
    mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

    FlushCommandQueue();
}

void BoxApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void BoxApp::OnMouseUp(WPARAM btnState, int x, int y)
{
    ReleaseCapture();
}

void BoxApp::OnMouseMove(WPARAM btnState, int x, int y)
{
    if ((btnState & MK_LBUTTON) != 0)
    {
        float dx = XMConvertToRadians(0.25f * static_cast<float>(x - mLastMousePos.x)) * mLookSpeed;
        float dy = XMConvertToRadians(0.25f * static_cast<float>(y - mLastMousePos.y)) * mLookSpeed;
        mCamera.Yaw(dx);
        mCamera.Pitch(dy);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
}

UINT BoxApp::CreateRgbaTexture(const std::string& name, UINT width, UINT height, const std::vector<std::uint8_t>& rgba)
{
    auto existing = mTextureIndexByName.find(name);
    if (existing != mTextureIndexByName.end())
        return existing->second;

    if (rgba.size() != static_cast<size_t>(width) * height * 4)
        throw std::runtime_error("RGBA texture data size does not match texture dimensions");

    auto tex = std::make_unique<Texture>();
    tex->Name = name;

    D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, width, height);
    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(tex->Resource.GetAddressOf())));

    const UINT64 uploadBufferSize = GetRequiredIntermediateSize(tex->Resource.Get(), 0, 1);
    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(tex->UploadHeap.GetAddressOf())));

    const D3D12_RESOURCE_STATES shaderState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    D3D12_SUBRESOURCE_DATA subresourceData = {};
    subresourceData.pData = rgba.data();
    subresourceData.RowPitch = static_cast<LONG_PTR>(width * 4);
    subresourceData.SlicePitch = subresourceData.RowPitch * height;
    UpdateSubresources(mCommandList.Get(), tex->Resource.Get(), tex->UploadHeap.Get(), 0, 0, 1, &subresourceData);
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        tex->Resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderState));

    const UINT newIndex = static_cast<UINT>(mTextures.size());
    mTextureIndexByName[name] = newIndex;
    mTextures.push_back(std::move(tex));
    return newIndex;
}

UINT BoxApp::CreateSolidColorTexture(const std::string& name, const std::array<std::uint8_t, 4>& rgba)
{
    return CreateRgbaTexture(name, 1, 1, std::vector<std::uint8_t>(rgba.begin(), rgba.end()));
}

UINT BoxApp::LoadOrCreateTexture(const std::filesystem::path& baseDir, const std::string& texName)
{
    if (texName.empty())
        return 0;

    const std::filesystem::path filePath = (baseDir / std::filesystem::path(texName)).lexically_normal();
    const std::string cacheKey = filePath.generic_string();

    auto it = mTextureIndexByName.find(cacheKey);
    if (it != mTextureIndexByName.end())
        return it->second;

    if (!std::filesystem::exists(filePath))
        throw std::runtime_error("Texture file not found: " + filePath.generic_string());

    auto tex = std::make_unique<Texture>();
    tex->Name = cacheKey;
    tex->Filename = filePath.wstring();

    const D3D12_RESOURCE_STATES shaderState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    const std::wstring extension = filePath.extension().wstring();

    if (extension == L".dds")
    {
        ThrowIfFailed(CreateDDSTextureFromFile12(
            md3dDevice.Get(),
            mCommandList.Get(),
            tex->Filename.c_str(),
            tex->Resource,
            tex->UploadHeap));

        mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
            tex->Resource.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, shaderState));
    }
    else if (extension == L".tga")
    {
        TgaImage img = LoadTgaRgba(tex->Filename);

        D3D12_RESOURCE_DESC texDesc = {};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Alignment = 0;
        texDesc.Width = static_cast<UINT>(img.Width);
        texDesc.Height = static_cast<UINT>(img.Height);
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 1;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.SampleDesc.Quality = 0;
        texDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        texDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

        ThrowIfFailed(md3dDevice->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            &texDesc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS(tex->Resource.GetAddressOf())));

        const UINT64 uploadBufferSize = GetRequiredIntermediateSize(tex->Resource.Get(), 0, 1);
        ThrowIfFailed(md3dDevice->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer(uploadBufferSize),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(tex->UploadHeap.GetAddressOf())));

        D3D12_SUBRESOURCE_DATA subresourceData = {};
        subresourceData.pData = img.Rgba.data();
        subresourceData.RowPitch = static_cast<LONG_PTR>(img.Width * 4);
        subresourceData.SlicePitch = subresourceData.RowPitch * img.Height;
        UpdateSubresources(mCommandList.Get(), tex->Resource.Get(), tex->UploadHeap.Get(), 0, 0, 1, &subresourceData);
        mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
            tex->Resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderState));
    }
    else
    {
        throw std::runtime_error("Unsupported texture format: " + filePath.generic_string());
    }

    const UINT newIndex = static_cast<UINT>(mTextures.size());
    mTextureIndexByName[cacheKey] = newIndex;
    mTextures.push_back(std::move(tex));
    return newIndex;
}

void BoxApp::BuildForestOctree()
{
    mForestOctreeRoot.reset();
    if (mForestObjects.empty())
        return;

    XMFLOAT3 minPoint(
        (std::numeric_limits<float>::max)(),
        (std::numeric_limits<float>::max)(),
        (std::numeric_limits<float>::max)());
    XMFLOAT3 maxPoint(
        -(std::numeric_limits<float>::max)(),
        -(std::numeric_limits<float>::max)(),
        -(std::numeric_limits<float>::max)());

    for (const CullObject& object : mForestObjects)
    {
        const XMFLOAT3& c = object.Bounds.Center;
        const XMFLOAT3& e = object.Bounds.Extents;

        minPoint.x = (std::min)(minPoint.x, c.x - e.x);
        minPoint.y = (std::min)(minPoint.y, c.y - e.y);
        minPoint.z = (std::min)(minPoint.z, c.z - e.z);

        maxPoint.x = (std::max)(maxPoint.x, c.x + e.x);
        maxPoint.y = (std::max)(maxPoint.y, c.y + e.y);
        maxPoint.z = (std::max)(maxPoint.z, c.z + e.z);
    }

    const float padding = 2.0f;
    minPoint.x -= padding;
    minPoint.y -= padding;
    minPoint.z -= padding;
    maxPoint.x += padding;
    maxPoint.y += padding;
    maxPoint.z += padding;

    const float centerY = 0.5f * (minPoint.y + maxPoint.y);
    constexpr float kForestOctreeVerticalHalfSize = 128.0f;
    minPoint.y = centerY - kForestOctreeVerticalHalfSize;
    maxPoint.y = centerY + kForestOctreeVerticalHalfSize;

    mForestOctreeRoot = std::make_unique<OctreeNode>();
    mForestOctreeRoot->Bounds = MakeBoundingBox(minPoint, maxPoint);
    mForestOctreeRoot->LooseBounds = mForestOctreeRoot->Bounds;

    for (UINT objectIndex = 0; objectIndex < static_cast<UINT>(mForestObjects.size()); ++objectIndex)
        InsertIntoOctree(*mForestOctreeRoot, objectIndex, 0);
}

void BoxApp::InsertIntoOctree(OctreeNode& node, UINT objectIndex, int depth)
{
    constexpr int kMaxDepth = 6;
    if (depth >= kMaxDepth)
    {
        node.ObjectIndices.push_back(objectIndex);
        return;
    }

    if (!node.Children[0])
    {
        const XMFLOAT3& c = node.Bounds.Center;
        const XMFLOAT3& e = node.Bounds.Extents;
        const XMFLOAT3 childExtents(0.5f * e.x, 0.5f * e.y, 0.5f * e.z);

        UINT childIndex = 0;
        for (int z = 0; z < 2; ++z)
        {
            for (int y = 0; y < 2; ++y)
            {
                for (int x = 0; x < 2; ++x)
                {
                    const float sx = x == 0 ? -1.0f : 1.0f;
                    const float sy = y == 0 ? -1.0f : 1.0f;
                    const float sz = z == 0 ? -1.0f : 1.0f;

                    auto child = std::make_unique<OctreeNode>();
                    const XMFLOAT3 childCenter(c.x + sx * childExtents.x, c.y + sy * childExtents.y, c.z + sz * childExtents.z);
                    child->Bounds = BoundingBox(
                        childCenter,
                        childExtents);
                    child->LooseBounds = BoundingBox(
                        childCenter,
                        XMFLOAT3(childExtents.x * 2.0f, childExtents.y * 2.0f, childExtents.z * 2.0f));
                    node.Children[childIndex++] = std::move(child);
                }
            }
        }
    }

    const BoundingBox& objectBounds = mForestObjects[objectIndex].Bounds;
    UINT childIndex = 0;
    childIndex |= objectBounds.Center.x >= node.Bounds.Center.x ? 1u : 0u;
    childIndex |= objectBounds.Center.y >= node.Bounds.Center.y ? 2u : 0u;
    childIndex |= objectBounds.Center.z >= node.Bounds.Center.z ? 4u : 0u;

    const auto& child = node.Children[childIndex];
    if (child && child->LooseBounds.Contains(objectBounds) == CONTAINS)
    {
        InsertIntoOctree(*child, objectIndex, depth + 1);
        return;
    }

    node.ObjectIndices.push_back(objectIndex);
}

void BoxApp::QueryOctree(const OctreeNode& node, const BoundingFrustum& frustum, const XMFLOAT3& cameraPos,
    std::vector<UINT>& visibleBatches, UINT& visibleObjects, UINT& visibleMeshObjects, UINT& visibleBillboardObjects) const
{
    const ContainmentType containment = frustum.Contains(node.LooseBounds);
    if (containment == DISJOINT)
        return;

    if (containment == CONTAINS)
    {
        CollectOctreeBatches(node, cameraPos, visibleBatches, visibleObjects, visibleMeshObjects, visibleBillboardObjects);
        return;
    }

    for (UINT objectIndex : node.ObjectIndices)
    {
        const CullObject& object = mForestObjects[objectIndex];
        if (frustum.Contains(object.Bounds) != DISJOINT)
        {
            AddVisibleTreeObject(objectIndex, cameraPos, visibleBatches, visibleObjects,
                visibleMeshObjects, visibleBillboardObjects);
        }
    }

    for (const auto& child : node.Children)
    {
        if (child)
            QueryOctree(*child, frustum, cameraPos, visibleBatches, visibleObjects,
                visibleMeshObjects, visibleBillboardObjects);
    }
}

void BoxApp::CollectOctreeBatches(const OctreeNode& node, const XMFLOAT3& cameraPos,
    std::vector<UINT>& visibleBatches, UINT& visibleObjects, UINT& visibleMeshObjects, UINT& visibleBillboardObjects) const
{
    for (UINT objectIndex : node.ObjectIndices)
    {
        AddVisibleTreeObject(objectIndex, cameraPos, visibleBatches, visibleObjects,
            visibleMeshObjects, visibleBillboardObjects);
    }

    for (const auto& child : node.Children)
    {
        if (child)
            CollectOctreeBatches(*child, cameraPos, visibleBatches, visibleObjects,
                visibleMeshObjects, visibleBillboardObjects);
    }
}

void BoxApp::BuildDescriptorHeaps()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = static_cast<UINT>(mTextures.size());
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mCbvSrvHeap)));

    UINT descriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_CPU_DESCRIPTOR_HANDLE hCpu(mCbvSrvHeap->GetCPUDescriptorHandleForHeapStart());

    for (auto& tex : mTextures)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format = tex->Resource->GetDesc().Format;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MostDetailedMip = 0;
        srvDesc.Texture2D.MipLevels = tex->Resource->GetDesc().MipLevels;
        srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;
        md3dDevice->CreateShaderResourceView(tex->Resource.Get(), &srvDesc, hCpu);
        hCpu.Offset(1, descriptorSize);
    }
}

void BoxApp::BuildDeferredSrvHeap()
{
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.NumDescriptors = 3;
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mDeferredSrvHeap)));
}

void BoxApp::UpdateDeferredSrvDescriptors()
{
    ID3D12Resource* albedo = mDeferredRenderer.Buffers.AlbedoResource();
    ID3D12Resource* normal = mDeferredRenderer.Buffers.NormalResource();
    ID3D12Resource* depth = mDepthStencilBuffer.Get();
    assert(albedo && normal && depth);

    auto dstCpu = mDeferredSrvHeap->GetCPUDescriptorHandleForHeapStart();
    const UINT descriptorSize = md3dDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    D3D12_SHADER_RESOURCE_VIEW_DESC albedoSrv = {};
    albedoSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    albedoSrv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    albedoSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    albedoSrv.Texture2D.MipLevels = 1;
    md3dDevice->CreateShaderResourceView(albedo, &albedoSrv, dstCpu);
    dstCpu.ptr += descriptorSize;

    D3D12_SHADER_RESOURCE_VIEW_DESC normalSrv = albedoSrv;
    normalSrv.Format = DXGI_FORMAT_R16G16_FLOAT;
    md3dDevice->CreateShaderResourceView(normal, &normalSrv, dstCpu);
    dstCpu.ptr += descriptorSize;

    D3D12_SHADER_RESOURCE_VIEW_DESC depthSrv = {};
    depthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    depthSrv.Format = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    depthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    depthSrv.Texture2D.MipLevels = 1;
    md3dDevice->CreateShaderResourceView(depth, &depthSrv, dstCpu);
}

void BoxApp::BuildConstantBuffers()
{
    const UINT objectCount = static_cast<UINT>(std::max<size_t>(1, mDrawBatches.size()));
    mObjectCB = std::make_unique<UploadBuffer<ObjectConstants>>(md3dDevice.Get(), objectCount, true);
    mDeferredCB = std::make_unique<UploadBuffer<DeferredPassConstants>>(md3dDevice.Get(), 1, true);
}

void BoxApp::BuildRootSignatures()
{
    CD3DX12_DESCRIPTOR_RANGE geomSrvTable[3];
    geomSrvTable[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    geomSrvTable[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1);
    geomSrvTable[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);

    CD3DX12_ROOT_PARAMETER geomRootParameter[4];
    geomRootParameter[0].InitAsConstantBufferView(0);
    geomRootParameter[1].InitAsDescriptorTable(1, &geomSrvTable[0], D3D12_SHADER_VISIBILITY_PIXEL);
    geomRootParameter[2].InitAsDescriptorTable(1, &geomSrvTable[1], D3D12_SHADER_VISIBILITY_PIXEL);
    geomRootParameter[3].InitAsDescriptorTable(1, &geomSrvTable[2], D3D12_SHADER_VISIBILITY_DOMAIN);

    CD3DX12_STATIC_SAMPLER_DESC linearWrap(0,
        D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP);

    CD3DX12_ROOT_SIGNATURE_DESC geomRootSigDesc(4, geomRootParameter, 1, &linearWrap,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> serializedRootSig = nullptr;
    ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&geomRootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
    if (errorBlob != nullptr)
        ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(0, serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(), IID_PPV_ARGS(&mGeometryRootSignature)));

    CD3DX12_DESCRIPTOR_RANGE lightSrvTable;
    lightSrvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0);

    CD3DX12_ROOT_PARAMETER lightRootParameter[2];
    lightRootParameter[0].InitAsDescriptorTable(1, &lightSrvTable, D3D12_SHADER_VISIBILITY_PIXEL);
    lightRootParameter[1].InitAsConstantBufferView(1);

    CD3DX12_ROOT_SIGNATURE_DESC lightRootSigDesc(2, lightRootParameter, 1, &linearWrap,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    serializedRootSig.Reset();
    errorBlob.Reset();
    hr = D3D12SerializeRootSignature(&lightRootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());
    if (errorBlob != nullptr)
        ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(0, serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(), IID_PPV_ARGS(&mLightingRootSignature)));
}

void BoxApp::BuildShadersAndInputLayout()
{
    mGBufferVS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "GBufferVS", "vs_5_0");
    mGBufferPS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "GBufferPS", "ps_5_0");
    mGBufferTessVS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "GBufferTessVS", "vs_5_0");
    mGBufferHS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "GBufferHS", "hs_5_0");
    mGBufferDS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "GBufferDS", "ds_5_0");
    mLightingVS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "LightingVS", "vs_5_0");
    mLightingPS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "LightingPS", "ps_5_0");
    mDebugWirePS = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "DebugTessWirePS", "ps_5_0");

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TANGENT", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 36, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };
}

void BoxApp::BuildLights()
{
    mDeferredRenderer.Lighting.PointLights.clear();
    mDeferredRenderer.Lighting.DirectionalLights.clear();
    mDeferredRenderer.Lighting.SpotLights.clear();

    mDeferredRenderer.Lighting.DirectionalLights.push_back({ XMFLOAT3(0.35f, -1.0f, 0.25f), 0.45f, XMFLOAT3(1.0f, 0.96f, 0.9f), 0.0f });

    mDeferredRenderer.Lighting.PointLights.push_back({ XMFLOAT3(-1.8f, 1.9f, -1.2f), 6.5f, XMFLOAT3(1.0f, 0.45f, 0.3f), 10.0f });
    mDeferredRenderer.Lighting.PointLights.push_back({ XMFLOAT3(1.9f, 2.0f, -0.3f), 6.5f, XMFLOAT3(0.25f, 0.55f, 1.0f), 10.0f });
    mDeferredRenderer.Lighting.PointLights.push_back({ XMFLOAT3(0.0f, 5.5f, 0.0f), 14.0f, XMFLOAT3(1.0f, 0.95f, 0.8f), 8.0f });

    SpotLight spotlight;
    spotlight.Position = XMFLOAT3(0.0f, 4.5f, -2.5f);
    spotlight.Direction = XMFLOAT3(0.0f, -0.9f, 0.35f);
    spotlight.InnerCos = 0.94f;
    spotlight.OuterCos = 0.84f;
    spotlight.Radius = 16.0f;
    spotlight.Intensity = 12.0f;
    spotlight.Color = XMFLOAT3(1.0f, 0.95f, 0.8f);
    mDeferredRenderer.Lighting.SpotLights.push_back(spotlight);
}

void BoxApp::BuildBoxGeometry()
{
    mTextures.clear();
    mTextureIndexByName.clear();
    mDrawBatches.clear();
    mVisibleBatchIndices.clear();
    mForestObjects.clear();
    mForestOctreeRoot.reset();

    const UINT whiteSrv = CreateSolidColorTexture("__white", { 255, 255, 255, 255 });
    const UINT flatNormalSrv = CreateSolidColorTexture("__flatNormal", { 128, 128, 255, 255 });
    const UINT neutralDisplacementSrv = CreateSolidColorTexture("__neutralDisplacement", { 128, 128, 128, 255 });

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    struct GeometryRange
    {
        UINT IndexCount = 0;
        UINT StartIndexLocation = 0;
        BoundingBox LocalBounds;
    };

    auto appendGeometry = [&](const std::vector<Vertex>& batchVertices, const std::vector<std::uint32_t>& batchIndices)
    {
        GeometryRange range;
        range.StartIndexLocation = static_cast<UINT>(indices.size());
        range.IndexCount = static_cast<UINT>(batchIndices.size());
        range.LocalBounds = ComputeLocalBounds(batchVertices);

        const std::uint32_t baseVertex = static_cast<std::uint32_t>(vertices.size());
        vertices.insert(vertices.end(), batchVertices.begin(), batchVertices.end());
        for (std::uint32_t idx : batchIndices)
            indices.push_back(baseVertex + idx);

        return range;
    };

    auto appendBatchGeometry = [&](const std::vector<Vertex>& batchVertices, const std::vector<std::uint32_t>& batchIndices, const DrawBatch& sourceBatch)
    {
        DrawBatch batch = sourceBatch;
        const GeometryRange range = appendGeometry(batchVertices, batchIndices);
        batch.StartIndexLocation = range.StartIndexLocation;
        batch.IndexCount = range.IndexCount;
        batch.Bounds = TransformBounds(range.LocalBounds, batch.World);

        const UINT batchIndex = static_cast<UINT>(mDrawBatches.size());
        mDrawBatches.push_back(batch);
        return batchIndex;
    };

    {
        std::string inputfile = "sponza-master\\sponza.obj";

        tinyobj::ObjReaderConfig reader_config;
        reader_config.triangulate = true;
        reader_config.mtl_search_path = "sponza-master\\";

        tinyobj::ObjReader reader;
        if (!reader.ParseFromFile(inputfile, reader_config))
        {
            if (!reader.Error().empty())
                OutputDebugStringA(reader.Error().c_str());

            throw std::runtime_error("Failed to load sponza.obj (tinyobj).");
        }

        const auto& attrib = reader.GetAttrib();
        const auto& shapes = reader.GetShapes();
        const auto& materials = reader.GetMaterials();
        const std::filesystem::path texBase = std::filesystem::path("sponza-master");

        struct BatchKey
        {
            int MaterialId = -1;
            size_t ShapeIndex = 0;

            bool operator==(const BatchKey& rhs) const
            {
                return MaterialId == rhs.MaterialId && ShapeIndex == rhs.ShapeIndex;
            }
        };

        struct BatchKeyHash
        {
            size_t operator()(const BatchKey& key) const
            {
                size_t h1 = std::hash<int>{}(key.MaterialId);
                size_t h2 = std::hash<size_t>{}(key.ShapeIndex);
                return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
            }
        };

        struct BatchBucket
        {
            DrawBatch Batch;
            std::vector<Vertex> Vertices;
            std::vector<std::uint32_t> LocalIndices;
        };

        std::unordered_map<BatchKey, BatchBucket, BatchKeyHash> buckets;
        std::vector<BatchKey> batchOrder;

        for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex)
        {
            const auto& shape = shapes[shapeIndex];
            size_t faceOffset = 0;

            for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f)
            {
                const int fv = shape.mesh.num_face_vertices[f];
                const int materialId = (f < shape.mesh.material_ids.size()) ? shape.mesh.material_ids[f] : -1;
                const BatchKey key{ materialId, shapeIndex };

                auto it = buckets.find(key);
                if (it == buckets.end())
                {
                    BatchBucket bucket;
                    bucket.Batch.DiffuseSrvIndex = whiteSrv;
                    bucket.Batch.NormalSrvIndex = flatNormalSrv;
                    bucket.Batch.DisplacementSrvIndex = neutralDisplacementSrv;

                    if (materialId >= 0 && materialId < static_cast<int>(materials.size()))
                    {
                        const auto& material = materials[materialId];
                        if (!material.diffuse_texname.empty())
                            bucket.Batch.DiffuseSrvIndex = LoadOrCreateTexture(texBase, material.diffuse_texname);

                        const std::string normalTexName = !material.normal_texname.empty() ? material.normal_texname :
                            (!material.bump_texname.empty() ? material.bump_texname : material.displacement_texname);
                        if (!normalTexName.empty())
                            bucket.Batch.NormalSrvIndex = LoadOrCreateTexture(texBase, normalTexName);

                        const std::string displacementTexName = !material.displacement_texname.empty() ? material.displacement_texname : normalTexName;
                        if (!displacementTexName.empty())
                            bucket.Batch.DisplacementSrvIndex = LoadOrCreateTexture(texBase, displacementTexName);

                        bucket.Batch.Tint = XMFLOAT4(material.diffuse[0], material.diffuse[1], material.diffuse[2], 1.0f);
                    }

                    auto inserted = buckets.emplace(key, std::move(bucket));
                    it = inserted.first;
                    batchOrder.push_back(key);
                }

                BatchBucket& bucket = it->second;
                for (int v = 0; v < fv; ++v)
                {
                    const auto& idx = shape.mesh.indices[faceOffset + v];
                    Vertex vert = {};
                    vert.Pos.x = attrib.vertices[3 * idx.vertex_index + 0] * 0.01f;
                    vert.Pos.y = attrib.vertices[3 * idx.vertex_index + 1] * 0.01f;
                    vert.Pos.z = attrib.vertices[3 * idx.vertex_index + 2] * 0.01f;

                    if (idx.normal_index >= 0 && !attrib.normals.empty())
                    {
                        vert.Normal.x = attrib.normals[3 * idx.normal_index + 0];
                        vert.Normal.y = attrib.normals[3 * idx.normal_index + 1];
                        vert.Normal.z = attrib.normals[3 * idx.normal_index + 2];
                    }
                    else
                    {
                        vert.Normal = XMFLOAT3(0.0f, 1.0f, 0.0f);
                    }

                    if (idx.texcoord_index >= 0 && !attrib.texcoords.empty())
                    {
                        vert.TexC.x = attrib.texcoords[2 * idx.texcoord_index + 0];
                        vert.TexC.y = 1.0f - attrib.texcoords[2 * idx.texcoord_index + 1];
                    }
                    else
                    {
                        vert.TexC = XMFLOAT2(0.0f, 0.0f);
                    }

                    vert.Tangent = BuildFallbackTangent(vert.Normal);

                    bucket.Vertices.push_back(vert);
                    bucket.LocalIndices.push_back(static_cast<uint32_t>(bucket.Vertices.size() - 1));
                }

                faceOffset += fv;
            }
        }

        for (const BatchKey& key : batchOrder)
        {
            BatchBucket& bucket = buckets[key];
            ComputeTangents(bucket.Vertices, bucket.LocalIndices);
            appendBatchGeometry(bucket.Vertices, bucket.LocalIndices, bucket.Batch);
        }
    }

    {
        GeometryGenerator geoGen;
        const std::filesystem::path texBase = std::filesystem::path("sponza-master");
        const std::filesystem::path texturesBase = std::filesystem::path("..") / ".." / "Textures";

        auto addGeneratedMesh = [&](const GeometryGenerator::MeshData& meshData, DrawBatch batch)
        {
            std::vector<Vertex> localVertices;
            localVertices.reserve(meshData.Vertices.size());

            for (const auto& srcVertex : meshData.Vertices)
            {
                Vertex v = {};
                v.Pos = srcVertex.Position;
                v.Normal = srcVertex.Normal;
                v.Tangent = srcVertex.TangentU;
                v.TexC = srcVertex.TexC;
                localVertices.push_back(v);
            }

            ComputeTangents(localVertices, meshData.Indices32);
            appendBatchGeometry(localVertices, meshData.Indices32, batch);
        };

        DrawBatch plinthBatch;
        plinthBatch.DiffuseSrvIndex = LoadOrCreateTexture(texBase, "textures/spnza_bricks_a_diff.tga");
        plinthBatch.NormalSrvIndex = LoadOrCreateTexture(texBase, "textures/spnza_bricks_a_ddn.tga");
        plinthBatch.DisplacementSrvIndex = LoadOrCreateTexture(texBase, "textures/spnza_bricks_a_ddn.tga");
        plinthBatch.Tessellated = true;
        plinthBatch.UvScale = XMFLOAT2(2.0f, 2.0f);
        plinthBatch.DisplacementScale = 0.08f;
        XMStoreFloat4x4(&plinthBatch.World, XMMatrixRotationY(0.35f) * XMMatrixTranslation(-1.8f, 0.8f, -1.3f));
        addGeneratedMesh(geoGen.CreateBox(1.5f, 1.6f, 1.5f, 0), plinthBatch);

        DrawBatch columnBatch;
        columnBatch.DiffuseSrvIndex = LoadOrCreateTexture(texBase, "textures/vase_dif.tga");
        columnBatch.NormalSrvIndex = LoadOrCreateTexture(texBase, "textures/vase_ddn.tga");
        columnBatch.DisplacementSrvIndex = LoadOrCreateTexture(texBase, "textures/vase_ddn.tga");
        columnBatch.Tessellated = true;
        columnBatch.UvScale = XMFLOAT2(1.0f, 1.6f);
        columnBatch.DisplacementScale = 0.06f;
        XMStoreFloat4x4(&columnBatch.World, XMMatrixRotationY(-0.45f) * XMMatrixTranslation(1.8f, 1.2f, -0.5f));
        addGeneratedMesh(geoGen.CreateCylinder(0.55f, 0.85f, 2.4f, 24, 6), columnBatch);

        DrawBatch waterBatch;
        waterBatch.DiffuseSrvIndex = LoadOrCreateTexture(texturesBase, "water1.dds");
        waterBatch.NormalSrvIndex = LoadOrCreateTexture(texturesBase, "default_nmap.dds");
        waterBatch.DisplacementSrvIndex = LoadOrCreateTexture(texturesBase, "water1.dds");
        waterBatch.Tessellated = true;
        waterBatch.AnimateUv = true;
        waterBatch.UvScale = XMFLOAT2(3.0f, 3.0f);
        waterBatch.DisplacementScale = 0.08f;
        waterBatch.Tint = XMFLOAT4(0.72f, 0.86f, 0.98f, 1.0f);
        XMStoreFloat4x4(&waterBatch.World, XMMatrixTranslation(-6.65f, 0.06f, -0.45f));
        addGeneratedMesh(geoGen.CreateGrid(6.0f, 4.0f, 8, 8), waterBatch);

        constexpr int kTreeColumns = 200;
        constexpr int kTreeRows = 100;
        constexpr UINT kTreeCount = kTreeColumns * kTreeRows;
        constexpr float kTreeSpacing = 1.65f;
        constexpr float kForestPadding = 16.0f;
        constexpr float kForestWidth = static_cast<float>(kTreeColumns - 1) * kTreeSpacing + kForestPadding;
        constexpr float kForestDepth = static_cast<float>(kTreeRows - 1) * kTreeSpacing + kForestPadding;

        DrawBatch fieldBatch;
        fieldBatch.SceneId = kSceneForest;
        fieldBatch.DiffuseSrvIndex = LoadOrCreateTexture(texturesBase, "grass.dds");
        fieldBatch.NormalSrvIndex = LoadOrCreateTexture(texturesBase, "default_nmap.dds");
        fieldBatch.DisplacementSrvIndex = neutralDisplacementSrv;
        fieldBatch.UvScale = XMFLOAT2(kForestWidth * 0.2f, kForestDepth * 0.2f);
        fieldBatch.Tint = XMFLOAT4(0.72f, 0.92f, 0.58f, 1.0f);
        addGeneratedMesh(geoGen.CreateGrid(kForestWidth, kForestDepth, 96, 96), fieldBatch);

        const UINT treeAtlasSrv = CreateRgbaTexture("__treeAtlas", 2, 1,
            std::vector<std::uint8_t>
            {
                112, 73, 42, 255,
                35, 115, 47, 255
            });
        const UINT treeBillboardSrv = LoadOrCreateTexture(texturesBase, "tree01S.dds");

        std::vector<Vertex> treeVertices;
        std::vector<std::uint32_t> treeIndices;

        auto appendTreeMesh = [&](const GeometryGenerator::MeshData& meshData, FXMMATRIX transform, float atlasU)
        {
            const std::uint32_t baseVertex = static_cast<std::uint32_t>(treeVertices.size());
            treeVertices.reserve(treeVertices.size() + meshData.Vertices.size());

            for (const auto& srcVertex : meshData.Vertices)
            {
                Vertex v = {};
                XMStoreFloat3(&v.Pos, XMVector3Transform(XMLoadFloat3(&srcVertex.Position), transform));
                XMStoreFloat3(&v.Normal, XMVector3Normalize(XMVector3TransformNormal(XMLoadFloat3(&srcVertex.Normal), transform)));
                XMStoreFloat3(&v.Tangent, XMVector3Normalize(XMVector3TransformNormal(XMLoadFloat3(&srcVertex.TangentU), transform)));
                v.TexC = XMFLOAT2(atlasU, 0.5f);
                treeVertices.push_back(v);
            }

            treeIndices.reserve(treeIndices.size() + meshData.Indices32.size());
            for (std::uint32_t index : meshData.Indices32)
                treeIndices.push_back(baseVertex + index);
        };

        appendTreeMesh(geoGen.CreateCylinder(0.10f, 0.08f, 0.95f, 6, 1), XMMatrixTranslation(0.0f, 0.475f, 0.0f), 0.25f);
        appendTreeMesh(geoGen.CreateCylinder(0.78f, 0.14f, 1.25f, 8, 1), XMMatrixTranslation(0.0f, 1.35f, 0.0f), 0.75f);
        appendTreeMesh(geoGen.CreateCylinder(0.56f, 0.04f, 0.95f, 8, 1), XMMatrixTranslation(0.0f, 2.00f, 0.0f), 0.75f);

        const GeometryRange treeRange = appendGeometry(treeVertices, treeIndices);

        const std::vector<Vertex> billboardVertices =
        {
            { XMFLOAT3(-0.5f, 0.0f, 0.0f), XMFLOAT3(0.0f, 0.0f, 1.0f), XMFLOAT3(1.0f, 0.0f, 0.0f), XMFLOAT2(0.0f, 1.0f) },
            { XMFLOAT3(-0.5f, 1.0f, 0.0f), XMFLOAT3(0.0f, 0.0f, 1.0f), XMFLOAT3(1.0f, 0.0f, 0.0f), XMFLOAT2(0.0f, 0.0f) },
            { XMFLOAT3(0.5f, 1.0f, 0.0f), XMFLOAT3(0.0f, 0.0f, 1.0f), XMFLOAT3(1.0f, 0.0f, 0.0f), XMFLOAT2(1.0f, 0.0f) },
            { XMFLOAT3(0.5f, 0.0f, 0.0f), XMFLOAT3(0.0f, 0.0f, 1.0f), XMFLOAT3(1.0f, 0.0f, 0.0f), XMFLOAT2(1.0f, 1.0f) }
        };
        const std::vector<std::uint32_t> billboardIndices = { 0, 1, 2, 0, 2, 3 };
        const GeometryRange billboardRange = appendGeometry(billboardVertices, billboardIndices);

        std::mt19937 rng(20260425u);
        std::uniform_real_distribution<float> jitter(-0.46f, 0.46f);
        std::uniform_real_distribution<float> treeScale(0.78f, 1.24f);
        std::uniform_real_distribution<float> treeHeight(0.86f, 1.48f);
        std::uniform_real_distribution<float> treeRotation(0.0f, MathHelper::Pi * 2.0f);

        mDrawBatches.reserve(mDrawBatches.size() + kTreeCount * 2);
        mForestObjects.reserve(kTreeCount);

        const float halfForestX = 0.5f * static_cast<float>(kTreeColumns - 1) * kTreeSpacing;
        const float halfForestZ = 0.5f * static_cast<float>(kTreeRows - 1) * kTreeSpacing;

        for (int z = 0; z < kTreeRows; ++z)
        {
            for (int x = 0; x < kTreeColumns; ++x)
            {
                const float px = static_cast<float>(x) * kTreeSpacing - halfForestX + jitter(rng);
                const float pz = static_cast<float>(z) * kTreeSpacing - halfForestZ + jitter(rng);
                const float sxz = treeScale(rng);
                const float sy = treeHeight(rng);
                const float treeWorldWidth = treeRange.LocalBounds.Extents.x * 2.0f * sxz;
                const float treeWorldHeight = (treeRange.LocalBounds.Center.y + treeRange.LocalBounds.Extents.y) * sy;
                const float billboardWidth = treeWorldWidth * 1.08f;
                const float billboardHeight = treeWorldHeight * 1.04f;
                const XMFLOAT3 billboardBase(px, 0.0f, pz);
                const UINT cullObjectIndex = static_cast<UINT>(mForestObjects.size());

                DrawBatch treeBatch;
                treeBatch.IndexCount = treeRange.IndexCount;
                treeBatch.StartIndexLocation = treeRange.StartIndexLocation;
                treeBatch.DiffuseSrvIndex = treeAtlasSrv;
                treeBatch.NormalSrvIndex = flatNormalSrv;
                treeBatch.DisplacementSrvIndex = neutralDisplacementSrv;
                treeBatch.SceneId = kSceneForest;
                treeBatch.Cullable = true;
                treeBatch.CullObjectIndex = cullObjectIndex;

                XMStoreFloat4x4(&treeBatch.World,
                    XMMatrixScaling(sxz, sy, sxz) *
                    XMMatrixRotationY(treeRotation(rng)) *
                    XMMatrixTranslation(px, 0.0f, pz));
                treeBatch.Bounds = TransformBounds(treeRange.LocalBounds, treeBatch.World);

                DrawBatch billboardBatch;
                billboardBatch.IndexCount = billboardRange.IndexCount;
                billboardBatch.StartIndexLocation = billboardRange.StartIndexLocation;
                billboardBatch.DiffuseSrvIndex = treeBillboardSrv;
                billboardBatch.NormalSrvIndex = flatNormalSrv;
                billboardBatch.DisplacementSrvIndex = neutralDisplacementSrv;
                billboardBatch.SceneId = kSceneForest;
                billboardBatch.Cullable = true;
                billboardBatch.Billboard = true;
                billboardBatch.CullObjectIndex = cullObjectIndex;
                billboardBatch.BillboardWidth = billboardWidth;
                billboardBatch.BillboardHeight = billboardHeight;
                billboardBatch.BillboardBase = billboardBase;
                XMStoreFloat4x4(&billboardBatch.World,
                    XMMatrixScaling(billboardWidth, billboardHeight, 1.0f) *
                    XMMatrixTranslation(px, 0.0f, pz));
                billboardBatch.Bounds = BoundingBox(
                    XMFLOAT3(px, billboardHeight * 0.5f, pz),
                    XMFLOAT3(billboardWidth * 0.5f, billboardHeight * 0.5f, billboardWidth * 0.5f));

                const BoundingBox cullBounds = MergeBounds(treeBatch.Bounds, billboardBatch.Bounds);
                const UINT meshBatchIndex = static_cast<UINT>(mDrawBatches.size());
                mDrawBatches.push_back(treeBatch);
                const UINT billboardBatchIndex = static_cast<UINT>(mDrawBatches.size());
                mDrawBatches.push_back(billboardBatch);
                mForestObjects.push_back({ cullBounds, meshBatchIndex, billboardBatchIndex, billboardBase, billboardWidth, billboardHeight });
            }
        }
    }

    BuildForestOctree();

    const UINT vbByteSize = static_cast<UINT>(vertices.size() * sizeof(Vertex));
    const UINT ibByteSize = static_cast<UINT>(indices.size() * sizeof(std::uint32_t));

    mBoxGeo = std::make_unique<MeshGeometry>();
    mBoxGeo->Name = "sceneGeo";

    ThrowIfFailed(D3DCreateBlob(vbByteSize, &mBoxGeo->VertexBufferCPU));
    CopyMemory(mBoxGeo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

    ThrowIfFailed(D3DCreateBlob(ibByteSize, &mBoxGeo->IndexBufferCPU));
    CopyMemory(mBoxGeo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

    mBoxGeo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(
        md3dDevice.Get(), mCommandList.Get(),
        vertices.data(), vbByteSize,
        mBoxGeo->VertexBufferUploader);

    mBoxGeo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(
        md3dDevice.Get(), mCommandList.Get(),
        indices.data(), ibByteSize,
        mBoxGeo->IndexBufferUploader);

    mBoxGeo->VertexByteStride = sizeof(Vertex);
    mBoxGeo->VertexBufferByteSize = vbByteSize;
    mBoxGeo->IndexFormat = DXGI_FORMAT_R32_UINT;
    mBoxGeo->IndexBufferByteSize = ibByteSize;
}

void BoxApp::BuildPSOs()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gbufferPsoDesc = {};
    gbufferPsoDesc.InputLayout = { mInputLayout.data(), static_cast<UINT>(mInputLayout.size()) };
    gbufferPsoDesc.pRootSignature = mGeometryRootSignature.Get();
    gbufferPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mGBufferVS->GetBufferPointer()),
        mGBufferVS->GetBufferSize()
    };
    gbufferPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mGBufferPS->GetBufferPointer()),
        mGBufferPS->GetBufferSize()
    };
    gbufferPsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    gbufferPsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    gbufferPsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    gbufferPsoDesc.SampleMask = UINT_MAX;
    gbufferPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gbufferPsoDesc.NumRenderTargets = 2;
    gbufferPsoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gbufferPsoDesc.RTVFormats[1] = DXGI_FORMAT_R16G16_FLOAT;
    gbufferPsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    gbufferPsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    gbufferPsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&gbufferPsoDesc, IID_PPV_ARGS(&mGBufferPSO)));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC billboardPsoDesc = gbufferPsoDesc;
    billboardPsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&billboardPsoDesc, IID_PPV_ARGS(&mGBufferBillboardPSO)));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC gbufferTessPsoDesc = gbufferPsoDesc;
    gbufferTessPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mGBufferTessVS->GetBufferPointer()),
        mGBufferTessVS->GetBufferSize()
    };
    gbufferTessPsoDesc.HS =
    {
        reinterpret_cast<BYTE*>(mGBufferHS->GetBufferPointer()),
        mGBufferHS->GetBufferSize()
    };
    gbufferTessPsoDesc.DS =
    {
        reinterpret_cast<BYTE*>(mGBufferDS->GetBufferPointer()),
        mGBufferDS->GetBufferSize()
    };
    gbufferTessPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&gbufferTessPsoDesc, IID_PPV_ARGS(&mGBufferTessPSO)));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC tessWirePsoDesc = gbufferTessPsoDesc;
    tessWirePsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mDebugWirePS->GetBufferPointer()),
        mDebugWirePS->GetBufferSize()
    };
    tessWirePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    tessWirePsoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    tessWirePsoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    tessWirePsoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
    tessWirePsoDesc.NumRenderTargets = 1;
    tessWirePsoDesc.RTVFormats[0] = mBackBufferFormat;
    tessWirePsoDesc.RTVFormats[1] = DXGI_FORMAT_UNKNOWN;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&tessWirePsoDesc, IID_PPV_ARGS(&mTessWirePSO)));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC lightPsoDesc = {};
    lightPsoDesc.InputLayout = { nullptr, 0 };
    lightPsoDesc.pRootSignature = mLightingRootSignature.Get();
    lightPsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mLightingVS->GetBufferPointer()),
        mLightingVS->GetBufferSize()
    };
    lightPsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mLightingPS->GetBufferPointer()),
        mLightingPS->GetBufferSize()
    };
    lightPsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    lightPsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    lightPsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    lightPsoDesc.DepthStencilState.DepthEnable = FALSE;
    lightPsoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    lightPsoDesc.SampleMask = UINT_MAX;
    lightPsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    lightPsoDesc.NumRenderTargets = 1;
    lightPsoDesc.RTVFormats[0] = mBackBufferFormat;
    lightPsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    lightPsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    lightPsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&lightPsoDesc, IID_PPV_ARGS(&mLightingPSO)));
}
