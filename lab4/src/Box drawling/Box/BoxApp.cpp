#include "../../Common/d3dApp.h"
#include "../../Common/MathHelper.h"
#include "../../Common/UploadBuffer.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <cstdio>

using Microsoft::WRL::ComPtr;
using namespace DirectX;
using namespace DirectX::PackedVector;

// ---------------------------------------------------------------------------
// Vertex: position + normal + UV texture coordinate
// ---------------------------------------------------------------------------
struct Vertex
{
    XMFLOAT3 Pos;
    XMFLOAT3 Normal;
    XMFLOAT2 TexC;
};

// ---------------------------------------------------------------------------
// Per-object constant buffer (must match cbuffer layout in color.hlsl)
// Total: 240 bytes → aligned to 256 for D3D12 CB requirements.
// ---------------------------------------------------------------------------
struct ObjectConstants
{
    XMFLOAT4X4 WorldViewProj = MathHelper::Identity4x4(); //  64 bytes
    XMFLOAT4X4 World         = MathHelper::Identity4x4(); //  64 bytes
    XMFLOAT4X4 TexTransform  = MathHelper::Identity4x4(); //  64 bytes
    XMFLOAT4   DiffuseAlbedo = { 1, 1, 1, 1 };            //  16 bytes
    XMFLOAT3   LightDir      = { 0, 1, 0 };               //  12 bytes  (set in Update)
    float      Pad0          = 0;                          //   4 bytes
    XMFLOAT3   LightColor    = { 1, 1, 1 };               //  12 bytes
    float      Time          = 0;                          //   4 bytes
};                                                         // = 240 bytes

// ---------------------------------------------------------------------------
// Per-material draw record
// ---------------------------------------------------------------------------
struct MaterialDraw
{
    SubmeshGeometry submesh;
    int             textureIndex  = 0;
    XMFLOAT4        diffuseAlbedo = { 1, 1, 1, 1 };

    // UV tiling scale (multiplied onto UV in VS via TexTransform)
    float tilingX  = 1.0f;
    float tilingY  = 1.0f;

    // UV scroll animation
    bool  animated = false;
    float scrollX  = 0.0f;   // units per second along U
    float scrollY  = 0.0f;   // units per second along V
};

// ===========================================================================
//  TGA loader helpers (file-local, no header needed)
// ===========================================================================

// Load a 24- or 32-bit uncompressed (type 2) or RLE (type 10) TGA file.
// Converts BGR(A) → RGBA and flips rows to top-down for D3D12.
static bool LoadTGABytes(const char* path,
                          std::vector<uint8_t>& outRGBA,
                          UINT& outW, UINT& outH)
{
    FILE* fp = nullptr;
    if (fopen_s(&fp, path, "rb") != 0 || !fp) return false;

#pragma pack(push, 1)
    struct TGAHeader
    {
        uint8_t  idLen;
        uint8_t  colMapType;
        uint8_t  imgType;
        uint8_t  colMapSpec[5];
        uint16_t xOrigin, yOrigin;
        uint16_t width, height;
        uint8_t  bpp;
        uint8_t  imgDesc;
    };
#pragma pack(pop)

    TGAHeader hdr = {};
    fread(&hdr, sizeof(hdr), 1, fp);
    if (hdr.idLen) fseek(fp, hdr.idLen, SEEK_CUR);

    outW = hdr.width;
    outH = hdr.height;

    const int ch  = hdr.bpp / 8;   // bytes per pixel (3 or 4)
    const int num = hdr.width * hdr.height;

    outRGBA.resize(num * 4);

    // Convert one BGR(A) pixel to RGBA at destination index dst.
    auto conv = [&](const uint8_t* src, int dst)
    {
        outRGBA[dst * 4 + 0] = src[2];
        outRGBA[dst * 4 + 1] = src[1];
        outRGBA[dst * 4 + 2] = src[0];
        outRGBA[dst * 4 + 3] = (ch == 4) ? src[3] : 255u;
    };

    std::vector<uint8_t> px(ch);

    if (hdr.imgType == 2)   // uncompressed true-colour
    {
        for (int i = 0; i < num; i++)
        {
            fread(px.data(), 1, ch, fp);
            conv(px.data(), i);
        }
    }
    else if (hdr.imgType == 10) // RLE true-colour
    {
        int i = 0;
        while (i < num)
        {
            uint8_t pkt; fread(&pkt, 1, 1, fp);
            int cnt = (pkt & 0x7F) + 1;
            if (pkt & 0x80)
            {
                fread(px.data(), 1, ch, fp);
                for (int k = 0; k < cnt; k++) conv(px.data(), i++);
            }
            else
            {
                for (int k = 0; k < cnt; k++)
                {
                    fread(px.data(), 1, ch, fp);
                    conv(px.data(), i++);
                }
            }
        }
    }
    else
    {
        fclose(fp);
        return false; // unsupported TGA image type
    }

    // TGA is stored bottom-up unless imgDesc bit5 is set; flip for D3D12 (top-down).
    if ((hdr.imgDesc & 0x20) == 0)
    {
        for (UINT r = 0; r < hdr.height / 2; r++)
        {
            uint8_t* a = &outRGBA[r                    * hdr.width * 4];
            uint8_t* b = &outRGBA[(hdr.height - 1 - r) * hdr.width * 4];
            std::swap_ranges(a, a + hdr.width * 4, b);
        }
    }

    fclose(fp);
    return true;
}

// Create a D3D12 DXGI_FORMAT_R8G8B8A8_UNORM Texture2D from raw RGBA bytes
// and record the copy commands into cmdList.
// outUpload must stay alive until the GPU upload fence is signalled.
static void CreateTex2DFromRGBA(ID3D12Device*              device,
                                 ID3D12GraphicsCommandList* cmdList,
                                 const std::vector<uint8_t>& rgba,
                                 UINT w, UINT h,
                                 ComPtr<ID3D12Resource>&    outResource,
                                 ComPtr<ID3D12Resource>&    outUpload)
{
    D3D12_RESOURCE_DESC td = {};
    td.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width            = w;
    td.Height           = h;
    td.DepthOrArraySize = 1;
    td.MipLevels        = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc       = { 1, 0 };

    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE, &td,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&outResource)));

    UINT64 uploadSize = 0;
    device->GetCopyableFootprints(&td, 0, 1, 0, nullptr, nullptr, nullptr, &uploadSize);

    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer(uploadSize),
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&outUpload)));

    D3D12_SUBRESOURCE_DATA sub = {};
    sub.pData      = rgba.data();
    sub.RowPitch   = (LONG_PTR)(w * 4);
    sub.SlicePitch = (LONG_PTR)(w * h * 4);

    UpdateSubresources(cmdList, outResource.Get(), outUpload.Get(), 0, 0, 1, &sub);

    cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        outResource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
}

// ===========================================================================
//  BoxApp
// ===========================================================================
class BoxApp : public D3DApp
{
public:
    BoxApp(HINSTANCE hInstance);
    BoxApp(const BoxApp&) = delete;
    BoxApp& operator=(const BoxApp&) = delete;
    ~BoxApp();

    virtual bool Initialize() override;

private:
    virtual void OnResize()  override;
    virtual void Update(const GameTimer& gt) override;
    virtual void Draw(const GameTimer& gt)   override;

    virtual void OnMouseDown(WPARAM btnState, int x, int y) override;
    virtual void OnMouseUp  (WPARAM btnState, int x, int y) override;
    virtual void OnMouseMove(WPARAM btnState, int x, int y) override;

    void LoadTextures();
    void BuildBoxGeometry();
    void BuildDescriptorHeaps();
    void BuildConstantBuffers();
    void BuildTextureViews();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildPSO();

private:
    ComPtr<ID3D12RootSignature>  mRootSignature = nullptr;
    ComPtr<ID3D12PipelineState>  mPSO           = nullptr;
    ComPtr<ID3DBlob>             mvsByteCode    = nullptr;
    ComPtr<ID3DBlob>             mpsByteCode    = nullptr;

    // Combined CBV + SRV descriptor heap
    ComPtr<ID3D12DescriptorHeap> mCbvSrvHeap = nullptr;

    std::unique_ptr<UploadBuffer<ObjectConstants>> mObjectCB = nullptr;
    std::unique_ptr<MeshGeometry>                  mBoxGeo   = nullptr;

    std::vector<std::unique_ptr<Texture>> mTextures;       // loaded TGA textures
    std::vector<MaterialDraw>             mMaterialDraws;  // per-material draw calls

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

    XMFLOAT4X4 mWorld = MathHelper::Identity4x4();
    XMFLOAT4X4 mView  = MathHelper::Identity4x4();
    XMFLOAT4X4 mProj  = MathHelper::Identity4x4();

    // Camera (spherical).  mRadius tuned for a ~1-unit head model.
    float mTheta  = 1.5f * XM_PI;
    float mPhi    = XM_PIDIV4;
    float mRadius = 3.0f;

    POINT mLastMousePos;
};

// ===========================================================================
//  WinMain
// ===========================================================================
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

BoxApp::BoxApp(HINSTANCE hInstance) : D3DApp(hInstance) {}
BoxApp::~BoxApp() {}

// ===========================================================================
//  Initialize
// ===========================================================================
bool BoxApp::Initialize()
{
    if (!D3DApp::Initialize()) return false;
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    LoadTextures();       // upload TGA files to GPU (needs open cmd list)
    BuildBoxGeometry();   // load OBJ, build VB/IB, group by material
    BuildDescriptorHeaps();
    BuildConstantBuffers();
    BuildTextureViews();
    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildPSO();

    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmds[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmds), cmds);
    FlushCommandQueue();

    return true;
}

// ===========================================================================
//  OnResize
// ===========================================================================
void BoxApp::OnResize()
{
    D3DApp::OnResize();
    XMMATRIX P = XMMatrixPerspectiveFovLH(0.25f * MathHelper::Pi, AspectRatio(), 0.1f, 100.0f);
    XMStoreFloat4x4(&mProj, P);
}

// ===========================================================================
//  Update  –  rebuild per-material constant buffers
// ===========================================================================
void BoxApp::Update(const GameTimer& gt)
{
    float x = mRadius * sinf(mPhi) * cosf(mTheta);
    float z = mRadius * sinf(mPhi) * sinf(mTheta);
    float y = mRadius * cosf(mPhi);

    XMMATRIX view = XMMatrixLookAtLH(
        XMVectorSet(x, y, z, 1.0f),
        XMVectorZero(),
        XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));
    XMStoreFloat4x4(&mView, view);

    XMMATRIX world        = XMLoadFloat4x4(&mWorld);
    XMMATRIX proj         = XMLoadFloat4x4(&mProj);
    XMMATRIX worldViewProj = world * view * proj;

    // Key light: upper-left-front, warm white
    XMVECTOR lightDir = XMVector3Normalize(XMVectorSet(0.5f, 0.8f, -0.4f, 0.0f));
    XMFLOAT3 lightDirF3, lightColorF3 = { 1.0f, 0.98f, 0.92f };
    XMStoreFloat3(&lightDirF3, lightDir);

    for (int i = 0; i < (int)mMaterialDraws.size(); i++)
    {
        const MaterialDraw& mat = mMaterialDraws[i];

        ObjectConstants obj;
        XMStoreFloat4x4(&obj.WorldViewProj, XMMatrixTranspose(worldViewProj));
        XMStoreFloat4x4(&obj.World,         XMMatrixTranspose(world));
        // UV tiling + optional scroll animation
        XMMATRIX scale  = XMMatrixScaling(mat.tilingX, mat.tilingY, 1.0f);
        XMMATRIX scroll = XMMatrixIdentity();
        if (mat.animated)
            scroll = XMMatrixTranslation(mat.scrollX * gt.TotalTime(),
                                         mat.scrollY * gt.TotalTime(), 0.0f);
        XMStoreFloat4x4(&obj.TexTransform, XMMatrixTranspose(scale * scroll));

        obj.DiffuseAlbedo = mMaterialDraws[i].diffuseAlbedo;
        obj.LightDir      = lightDirF3;
        obj.LightColor    = lightColorF3;
        obj.Time = gt.TotalTime();

        mObjectCB->CopyData(i, obj);
    }
}

// ===========================================================================
//  Draw  –  one draw call per material group
// ===========================================================================
void BoxApp::Draw(const GameTimer& gt)
{
    ThrowIfFailed(mDirectCmdListAlloc->Reset());
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), mPSO.Get()));

    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

    mCommandList->ClearRenderTargetView(CurrentBackBufferView(), Colors::Black, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(),
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);

    mCommandList->OMSetRenderTargets(1, &CurrentBackBufferView(), true, &DepthStencilView());

    ID3D12DescriptorHeap* heaps[] = { mCbvSrvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(heaps), heaps);
    mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

    mCommandList->IASetVertexBuffers(0, 1, &mBoxGeo->VertexBufferView());
    mCommandList->IASetIndexBuffer(&mBoxGeo->IndexBufferView());
    mCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    const UINT descSize     = mCbvSrvUavDescriptorSize;
    const int  numMaterials = (int)mMaterialDraws.size();
    auto       gpuStart     = mCbvSrvHeap->GetGPUDescriptorHandleForHeapStart();

    // Descriptor layout:
    //   [0 .. numMaterials-1]         → CBV per material
    //   [numMaterials .. N+T-1]        → SRV per texture

    for (int i = 0; i < numMaterials; i++)
    {
        const auto& mat = mMaterialDraws[i];

        CD3DX12_GPU_DESCRIPTOR_HANDLE cbvHandle(gpuStart, i, descSize);
        mCommandList->SetGraphicsRootDescriptorTable(0, cbvHandle);

        int srvSlot = numMaterials + mat.textureIndex;
        CD3DX12_GPU_DESCRIPTOR_HANDLE srvHandle(gpuStart, srvSlot, descSize);
        mCommandList->SetGraphicsRootDescriptorTable(1, srvHandle);

        mCommandList->DrawIndexedInstanced(
            mat.submesh.IndexCount, 1,
            mat.submesh.StartIndexLocation,
            mat.submesh.BaseVertexLocation, 0);
    }

    mCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
        CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));

    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmds[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmds), cmds);

    ThrowIfFailed(mSwapChain->Present(0, 0));
    mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;
    FlushCommandQueue();
}

// ===========================================================================
//  Mouse input
// ===========================================================================
void BoxApp::OnMouseDown(WPARAM btnState, int x, int y)
{
    mLastMousePos = { x, y };
    SetCapture(mhMainWnd);
}

void BoxApp::OnMouseUp(WPARAM, int, int)
{
    ReleaseCapture();
}

void BoxApp::OnMouseMove(WPARAM btnState, int x, int y)
{
    if (btnState & MK_LBUTTON)
    {
        mTheta += XMConvertToRadians(0.25f * (x - mLastMousePos.x));
        mPhi   += XMConvertToRadians(0.25f * (y - mLastMousePos.y));
        mPhi    = MathHelper::Clamp(mPhi, 0.1f, MathHelper::Pi - 0.1f);
    }
    else if (btnState & MK_RBUTTON)
    {
        float d  = 0.005f * (float)(x - mLastMousePos.x - (y - mLastMousePos.y));
        mRadius += d;
        mRadius  = MathHelper::Clamp(mRadius, 0.5f, 8.0f);
    }
    mLastMousePos = { x, y };
}

// ===========================================================================
//  LoadTextures
//  Loads TGA files for the african_head model.
//
//  Required files next to african_head.obj (i.e. in the working directory):
//    african_head_diffuse.tga  – slot 0, used in the shader as t0
//    african_head_nm.tga       – slot 1, loaded and bound as t1 (ready for
//                                normal-mapping when the shader is extended)
// ===========================================================================
void BoxApp::LoadTextures()
{
    const char* paths[] =
    {
        "african_head_diffuse.tga",  // index 0 – diffuse colour map
        "african_head_nm.tga",       // index 1 – normal map (loaded, t1)
    };

    for (const char* path : paths)
    {
        std::vector<uint8_t> rgba;
        UINT w = 0, h = 0;
        if (!LoadTGABytes(path, rgba, w, h))
            throw std::runtime_error(std::string("Cannot load TGA: ") + path);

        auto tex = std::make_unique<Texture>();
        tex->Name     = path;      // used for matching with OBJ material names
        tex->Filename = AnsiToWString(path);

        CreateTex2DFromRGBA(md3dDevice.Get(), mCommandList.Get(),
                            rgba, w, h,
                            tex->Resource, tex->UploadHeap);

        mTextures.push_back(std::move(tex));
    }
}

// ===========================================================================
//  BuildBoxGeometry
//  Loads african_head.obj, reads normals and UVs, groups faces by material,
//  then uploads a single interleaved vertex + index buffer to the GPU.
// ===========================================================================
void BoxApp::BuildBoxGeometry()
{
    tinyobj::ObjReaderConfig cfg;
    cfg.triangulate = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile("african_head.obj", cfg))
    {
        if (!reader.Error().empty())
            OutputDebugStringA(reader.Error().c_str());
        throw std::runtime_error("Failed to load african_head.obj. "
                                 "Place the file in the working directory.");
    }

    const auto& attrib    = reader.GetAttrib();
    const auto& shapes    = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    // ---- Group faces by material ID ----------------------------------------
    struct MatGroup
    {
        std::vector<Vertex>   verts;
        std::vector<uint32_t> inds;
        int      texIdx = 0;
        XMFLOAT4 albedo = { 1, 1, 1, 1 };
    };
    std::map<int, MatGroup> groups;

    for (const auto& shape : shapes)
    {
        int idxOff = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int matId = (f < shape.mesh.material_ids.size())
                        ? shape.mesh.material_ids[f] : -1;
            if (matId < 0) matId = 0;

            auto& g = groups[matId];

            // On first use: resolve texture index and albedo from the MTL.
            if (g.verts.empty())
            {
                if (matId < (int)materials.size())
                {
                    // Match the MTL diffuse texture name to a loaded Texture.
                    const std::string& diffName = materials[matId].diffuse_texname;
                    for (int t = 0; t < (int)mTextures.size(); t++)
                    {
                        if (mTextures[t]->Name == diffName)
                        {
                            g.texIdx = t;
                            break;
                        }
                    }

                    // Albedo from MTL diffuse values (default to white if black).
                    const auto& m = materials[matId];
                    g.albedo =
                    {
                        m.diffuse[0] > 0.01f ? m.diffuse[0] : 1.0f,
                        m.diffuse[1] > 0.01f ? m.diffuse[1] : 1.0f,
                        m.diffuse[2] > 0.01f ? m.diffuse[2] : 1.0f,
                        1.0f
                    };
                }
            }

            // Build three vertices for this triangle.
            for (int v = 0; v < 3; v++)
            {
                const auto& idx = shape.mesh.indices[idxOff + v];

                Vertex vert = {};

                vert.Pos =
                {
                    attrib.vertices[3 * idx.vertex_index + 0],
                    attrib.vertices[3 * idx.vertex_index + 1],
                    attrib.vertices[3 * idx.vertex_index + 2],
                };
                // african_head is already in ~1-unit scale; no scaling needed.

                if (idx.normal_index >= 0)
                {
                    vert.Normal =
                    {
                        attrib.normals[3 * idx.normal_index + 0],
                        attrib.normals[3 * idx.normal_index + 1],
                        attrib.normals[3 * idx.normal_index + 2],
                    };
                }

                if (idx.texcoord_index >= 0)
                {
                    vert.TexC =
                    {
                        attrib.texcoords[2 * idx.texcoord_index + 0],
                        1.0f - attrib.texcoords[2 * idx.texcoord_index + 1], // flip V
                    };
                }

                g.inds.push_back((uint32_t)g.verts.size());
                g.verts.push_back(vert);
            }
            idxOff += 3;
        }
    }

    // ---- Flatten groups into a single VB / IB -----------------------------
    std::vector<Vertex>   allV;
    std::vector<uint32_t> allI;
    mMaterialDraws.clear();

    for (auto& kv : groups)
    {
        MatGroup& g = kv.second;
        if (g.inds.empty()) continue;

        MaterialDraw md;
        md.submesh.StartIndexLocation = (UINT)allI.size();
        md.submesh.BaseVertexLocation = (INT)allV.size();
        md.submesh.IndexCount         = (UINT)g.inds.size();
        md.textureIndex  = g.texIdx;
        md.diffuseAlbedo = g.albedo;

        // Tiling and animation per texture slot:
        //   slot 0 (diffuse) – 2×2 tiling + slow horizontal UV scroll
        //   slot 1 (normal)  – same tiling, no animation
        if (g.texIdx == 0)
        {
            md.tilingX  = 2.0f;
            md.tilingY  = 2.0f;
            md.animated = true;
            md.scrollX  = 0.04f;  // 0.04 UV units per second
            md.scrollY  = 0.0f;
        }
        else
        {
            md.tilingX  = 2.0f;
            md.tilingY  = 2.0f;
            md.animated = false;
        }

        allI.insert(allI.end(), g.inds.begin(), g.inds.end());
        allV.insert(allV.end(), g.verts.begin(), g.verts.end());

        mMaterialDraws.push_back(md);
    }

    // ---- Upload to GPU -----------------------------------------------------
    const UINT vbSize = (UINT)allV.size() * sizeof(Vertex);
    const UINT ibSize = (UINT)allI.size() * sizeof(uint32_t);

    mBoxGeo = std::make_unique<MeshGeometry>();
    mBoxGeo->Name = "headGeo";

    ThrowIfFailed(D3DCreateBlob(vbSize, &mBoxGeo->VertexBufferCPU));
    CopyMemory(mBoxGeo->VertexBufferCPU->GetBufferPointer(), allV.data(), vbSize);

    ThrowIfFailed(D3DCreateBlob(ibSize, &mBoxGeo->IndexBufferCPU));
    CopyMemory(mBoxGeo->IndexBufferCPU->GetBufferPointer(), allI.data(), ibSize);

    mBoxGeo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(
        md3dDevice.Get(), mCommandList.Get(), allV.data(), vbSize,
        mBoxGeo->VertexBufferUploader);

    mBoxGeo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(
        md3dDevice.Get(), mCommandList.Get(), allI.data(), ibSize,
        mBoxGeo->IndexBufferUploader);

    mBoxGeo->VertexByteStride     = sizeof(Vertex);
    mBoxGeo->VertexBufferByteSize = vbSize;
    mBoxGeo->IndexFormat          = DXGI_FORMAT_R32_UINT;
    mBoxGeo->IndexBufferByteSize  = ibSize;
}

// ===========================================================================
//  BuildDescriptorHeaps
//  Layout: [0..N-1] CBV per material | [N..N+T-1] SRV per texture
// ===========================================================================
void BoxApp::BuildDescriptorHeaps()
{
    const UINT total = (UINT)(mMaterialDraws.size() + mTextures.size());

    D3D12_DESCRIPTOR_HEAP_DESC desc = {};
    desc.NumDescriptors = total;
    desc.Type           = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.Flags          = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&mCbvSrvHeap)));
}

// ===========================================================================
//  BuildConstantBuffers
//  One CB element and one CBV descriptor per material.
// ===========================================================================
void BoxApp::BuildConstantBuffers()
{
    const UINT n = (UINT)mMaterialDraws.size();
    mObjectCB = std::make_unique<UploadBuffer<ObjectConstants>>(
        md3dDevice.Get(), n, true);

    const UINT cbSize   = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    const UINT descSize = mCbvSrvUavDescriptorSize;
    D3D12_GPU_VIRTUAL_ADDRESS base = mObjectCB->Resource()->GetGPUVirtualAddress();

    for (UINT i = 0; i < n; i++)
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = base + (UINT64)i * cbSize;
        cbvDesc.SizeInBytes    = cbSize;

        CD3DX12_CPU_DESCRIPTOR_HANDLE h(
            mCbvSrvHeap->GetCPUDescriptorHandleForHeapStart(), (INT)i, descSize);
        md3dDevice->CreateConstantBufferView(&cbvDesc, h);
    }
}

// ===========================================================================
//  BuildTextureViews
//  SRV descriptors placed after all CBV descriptors.
// ===========================================================================
void BoxApp::BuildTextureViews()
{
    const UINT numMat   = (UINT)mMaterialDraws.size();
    const UINT descSize = mCbvSrvUavDescriptorSize;

    for (UINT t = 0; t < (UINT)mTextures.size(); t++)
    {
        const auto& tex = mTextures[t];
        D3D12_RESOURCE_DESC rd = tex->Resource->GetDesc();

        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Shader4ComponentMapping         = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Format                          = rd.Format;
        srvDesc.ViewDimension                   = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MostDetailedMip       = 0;
        srvDesc.Texture2D.MipLevels             = rd.MipLevels;
        srvDesc.Texture2D.ResourceMinLODClamp   = 0.0f;

        CD3DX12_CPU_DESCRIPTOR_HANDLE h(
            mCbvSrvHeap->GetCPUDescriptorHandleForHeapStart(),
            (INT)(numMat + t), descSize);
        md3dDevice->CreateShaderResourceView(tex->Resource.Get(), &srvDesc, h);
    }
}

// ===========================================================================
//  BuildRootSignature
//  Slot 0: descriptor table → 1 CBV (b0)
//  Slot 1: descriptor table → 1 SRV (t0)
//  Static sampler s0: anisotropic wrap
// ===========================================================================
void BoxApp::BuildRootSignature()
{
    CD3DX12_DESCRIPTOR_RANGE cbvTable;
    cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);

    CD3DX12_DESCRIPTOR_RANGE srvTable;
    srvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);

    CD3DX12_ROOT_PARAMETER params[2];
    params[0].InitAsDescriptorTable(1, &cbvTable);
    params[1].InitAsDescriptorTable(1, &srvTable);

    CD3DX12_STATIC_SAMPLER_DESC sampler(
        0,
        D3D12_FILTER_ANISOTROPIC,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        0.0f, 8);

    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(
        2, params, 1, &sampler,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    ComPtr<ID3DBlob> blob, err;
    HRESULT hr = D3D12SerializeRootSignature(
        &rsDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(), err.GetAddressOf());
    if (err) ::OutputDebugStringA((char*)err->GetBufferPointer());
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0, blob->GetBufferPointer(), blob->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature)));
}

// ===========================================================================
//  BuildShadersAndInputLayout
// ===========================================================================
void BoxApp::BuildShadersAndInputLayout()
{
    mvsByteCode = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "VS", "vs_5_0");
    mpsByteCode = d3dUtil::CompileShader(L"Shaders\\color.hlsl", nullptr, "PS", "ps_5_0");

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

// ===========================================================================
//  BuildPSO
// ===========================================================================
void BoxApp::BuildPSO()
{
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout    = { mInputLayout.data(), (UINT)mInputLayout.size() };
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.VS = { (BYTE*)mvsByteCode->GetBufferPointer(), mvsByteCode->GetBufferSize() };
    psoDesc.PS = { (BYTE*)mpsByteCode->GetBufferPointer(), mpsByteCode->GetBufferSize() };
    psoDesc.RasterizerState   = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState        = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask        = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets  = 1;
    psoDesc.RTVFormats[0]     = mBackBufferFormat;
    psoDesc.SampleDesc.Count   = m4xMsaaState ? 4 : 1;
    psoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    psoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPSO)));
}
