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

using Microsoft::WRL::ComPtr;
using namespace DirectX;

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

    void BuildDescriptorHeaps();
    void BuildDeferredSrvHeap();
    void UpdateDeferredSrvDescriptors();
    void BuildConstantBuffers();
    void BuildRootSignatures();
    void BuildShadersAndInputLayout();
    void BuildBoxGeometry();
    void BuildPSOs();
    void BuildLights();
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
        bool Tessellated = false;
        XMFLOAT2 UvScale = XMFLOAT2(1.0f, 1.0f);
        float DisplacementScale = 0.0f;
        float Padding = 0.0f;
        XMFLOAT4 Tint = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
        XMFLOAT4X4 World = MathHelper::Identity4x4();
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
    std::vector<std::unique_ptr<Texture>> mTextures;
    std::unordered_map<std::string, UINT> mTextureIndexByName;

    ComPtr<ID3DBlob> mGBufferVS = nullptr;
    ComPtr<ID3DBlob> mGBufferPS = nullptr;
    ComPtr<ID3DBlob> mGBufferTessVS = nullptr;
    ComPtr<ID3DBlob> mGBufferHS = nullptr;
    ComPtr<ID3DBlob> mGBufferDS = nullptr;
    ComPtr<ID3DBlob> mLightingVS = nullptr;
    ComPtr<ID3DBlob> mLightingPS = nullptr;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

    ComPtr<ID3D12PipelineState> mGBufferPSO = nullptr;
    ComPtr<ID3D12PipelineState> mGBufferTessPSO = nullptr;
    ComPtr<ID3D12PipelineState> mLightingPSO = nullptr;

    FreeCamera mCamera;
    float mMoveSpeed = 10.0f;
    float mLookSpeed = 2.0f;

    UINT mDebugView = 0;
    bool mDebugViewKeyWasDown = false;
    float mTessMinDistance = 3.0f;
    float mTessMaxDistance = 20.0f;
    float mTessMinFactor = 1.0f;
    float mTessMaxFactor = 12.0f;

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

std::wstring BoxApp::GetAdditionalWindowText() const
{
    const XMFLOAT3 cameraPos = mCamera.GetPosition3f();
    static const std::array<const wchar_t*, 4> kDebugViewNames =
    {
        L"Lit",
        L"Albedo",
        L"Normal",
        L"Depth"
    };

    std::wostringstream stream;
    stream << std::fixed << std::setprecision(2)
           << L"cam xyz: (" << cameraPos.x << L", " << cameraPos.y << L", " << cameraPos.z << L")"
           << L" | debug: " << kDebugViewNames[mDebugView]
           << L" | tess range: [" << mTessMinDistance << L", " << mTessMaxDistance << L"]";

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

    const bool debugKeyDown = d3dUtil::IsKeyDown('G');
    if (debugKeyDown && !mDebugViewKeyWasDown)
        mDebugView = (mDebugView + 1) % 4;
    mDebugViewKeyWasDown = debugKeyDown;

    mCamera.UpdateViewMatrix();

    const XMMATRIX view = mCamera.GetView();
    const XMMATRIX proj = mCamera.GetProj();
    const XMMATRIX viewProj = view * proj;
    const XMFLOAT3 cameraPos = mCamera.GetPosition3f();

    for (size_t i = 0; i < mDrawBatches.size(); ++i)
    {
        const DrawBatch& batch = mDrawBatches[i];

        ObjectConstants objConstants;
        const XMMATRIX world = XMLoadFloat4x4(&batch.World);
        XMStoreFloat4x4(&objConstants.World, XMMatrixTranspose(world));
        XMStoreFloat4x4(&objConstants.ViewProj, XMMatrixTranspose(viewProj));
        objConstants.UvScale = batch.UvScale;
        objConstants.UvOffset = XMFLOAT2(0.0f, 0.0f);
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

    auto drawBatches = [&](bool tessellated, ID3D12PipelineState* pso, D3D_PRIMITIVE_TOPOLOGY topology)
    {
        mCommandList->SetPipelineState(pso);
        mCommandList->IASetPrimitiveTopology(topology);

        for (size_t i = 0; i < mDrawBatches.size(); ++i)
        {
            const DrawBatch& batch = mDrawBatches[i];
            if (batch.Tessellated != tessellated)
                continue;

            auto diffuseHandle = baseSrvHandle;
            diffuseHandle.ptr += static_cast<SIZE_T>(batch.DiffuseSrvIndex) * descSize;
            auto normalHandle = baseSrvHandle;
            normalHandle.ptr += static_cast<SIZE_T>(batch.NormalSrvIndex) * descSize;
            auto displacementHandle = baseSrvHandle;
            displacementHandle.ptr += static_cast<SIZE_T>(batch.DisplacementSrvIndex) * descSize;

            mCommandList->SetGraphicsRootConstantBufferView(0, objectCbAddress + static_cast<UINT64>(i) * objCBByteSize);
            mCommandList->SetGraphicsRootDescriptorTable(1, diffuseHandle);
            mCommandList->SetGraphicsRootDescriptorTable(2, normalHandle);
            mCommandList->SetGraphicsRootDescriptorTable(3, displacementHandle);
            mCommandList->DrawIndexedInstanced(batch.IndexCount, 1, batch.StartIndexLocation, 0, 0);
        }
    };

    drawBatches(false, mGBufferPSO.Get(), D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    drawBatches(true, mGBufferTessPSO.Get(), D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);

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

    CD3DX12_RESOURCE_BARRIER endFrame[2] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition(depth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE),
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

UINT BoxApp::CreateSolidColorTexture(const std::string& name, const std::array<std::uint8_t, 4>& rgba)
{
    auto existing = mTextureIndexByName.find(name);
    if (existing != mTextureIndexByName.end())
        return existing->second;

    auto tex = std::make_unique<Texture>();
    tex->Name = name;

    D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1);
    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &texDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(tex->Resource.GetAddressOf())));

    ThrowIfFailed(md3dDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(1024),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(tex->UploadHeap.GetAddressOf())));

    const D3D12_RESOURCE_STATES shaderState = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    D3D12_SUBRESOURCE_DATA subresourceData = {};
    subresourceData.pData = rgba.data();
    subresourceData.RowPitch = 4;
    subresourceData.SlicePitch = 4;
    UpdateSubresources(mCommandList.Get(), tex->Resource.Get(), tex->UploadHeap.Get(), 0, 0, 1, &subresourceData);
    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        tex->Resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST, shaderState));

    const UINT newIndex = static_cast<UINT>(mTextures.size());
    mTextureIndexByName[name] = newIndex;
    mTextures.push_back(std::move(tex));
    return newIndex;
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

    const UINT whiteSrv = CreateSolidColorTexture("__white", { 255, 255, 255, 255 });
    const UINT flatNormalSrv = CreateSolidColorTexture("__flatNormal", { 128, 128, 255, 255 });
    const UINT neutralDisplacementSrv = CreateSolidColorTexture("__neutralDisplacement", { 128, 128, 128, 255 });

    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    auto appendBatchGeometry = [&](const std::vector<Vertex>& batchVertices, const std::vector<std::uint32_t>& batchIndices, const DrawBatch& sourceBatch)
    {
        DrawBatch batch = sourceBatch;
        batch.StartIndexLocation = static_cast<UINT>(indices.size());
        batch.IndexCount = static_cast<UINT>(batchIndices.size());

        const std::uint32_t baseVertex = static_cast<std::uint32_t>(vertices.size());
        vertices.insert(vertices.end(), batchVertices.begin(), batchVertices.end());
        for (std::uint32_t idx : batchIndices)
            indices.push_back(baseVertex + idx);

        mDrawBatches.push_back(batch);
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
    }

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
