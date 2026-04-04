#include "GBuffer.h"

#include <algorithm>
#include <cassert>

using Microsoft::WRL::ComPtr;

bool GBuffer::IsInitialized() const
{
    return mRtvHeap != nullptr && mSrvHeap != nullptr;
}

void GBuffer::Build(ID3D12Device* device, UINT width, UINT height)
{
    mWidth = width;
    mHeight = height;

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = 2;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&mRtvHeap)));

    D3D12_DESCRIPTOR_HEAP_DESC srvHeapDesc = {};
    srvHeapDesc.NumDescriptors = 2;
    srvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&srvHeapDesc, IID_PPV_ARGS(&mSrvHeap)));

    CreateTextures(device);
    CreateViews(device);
}

void GBuffer::Resize(ID3D12Device* device, UINT width, UINT height)
{
    if (width == 0 || height == 0)
        return;
    if (!IsInitialized())
        return;

    mAlbedo.Reset();
    mNormalDepth.Reset();
    mWidth = width;
    mHeight = height;

    CreateTextures(device);
    CreateViews(device);
}

ID3D12DescriptorHeap* GBuffer::SrvHeap() const
{
    return mSrvHeap.Get();
}

ID3D12Resource* GBuffer::AlbedoResource() const
{
    return mAlbedo.Get();
}

ID3D12Resource* GBuffer::NormalDepthResource() const
{
    return mNormalDepth.Get();
}

D3D12_CPU_DESCRIPTOR_HANDLE GBuffer::AlbedoRtv() const
{
    return mAlbedoRtv;
}

D3D12_CPU_DESCRIPTOR_HANDLE GBuffer::NormalDepthRtv() const
{
    return mNormalDepthRtv;
}

void GBuffer::CreateTextures(ID3D12Device* device)
{
    auto albedoDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, mWidth, mHeight, 1, 1);
    auto normalDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R16G16B16A16_FLOAT, mWidth, mHeight, 1, 1);
    albedoDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    normalDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    const float black[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    D3D12_CLEAR_VALUE albedoClear = {};
    albedoClear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    std::copy(std::begin(black), std::end(black), std::begin(albedoClear.Color));

    D3D12_CLEAR_VALUE normalClear = {};
    normalClear.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    std::copy(std::begin(black), std::end(black), std::begin(normalClear.Color));

    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &albedoDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        &albedoClear,
        IID_PPV_ARGS(&mAlbedo)));

    ThrowIfFailed(device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &normalDesc,
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        &normalClear,
        IID_PPV_ARGS(&mNormalDepth)));
}

void GBuffer::CreateViews(ID3D12Device* device)
{
    assert(mRtvHeap && mSrvHeap);

    UINT rtvSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    mAlbedoRtv = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
    mNormalDepthRtv = mAlbedoRtv;
    mNormalDepthRtv.ptr += rtvSize;

    device->CreateRenderTargetView(mAlbedo.Get(), nullptr, mAlbedoRtv);
    device->CreateRenderTargetView(mNormalDepth.Get(), nullptr, mNormalDepthRtv);

    auto srvCpu = mSrvHeap->GetCPUDescriptorHandleForHeapStart();
    UINT srvSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_SHADER_RESOURCE_VIEW_DESC albedoSrv = {};
    albedoSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    albedoSrv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    albedoSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    albedoSrv.Texture2D.MipLevels = 1;

    D3D12_SHADER_RESOURCE_VIEW_DESC normalSrv = albedoSrv;
    normalSrv.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;

    device->CreateShaderResourceView(mAlbedo.Get(), &albedoSrv, srvCpu);
    srvCpu.ptr += srvSize;
    device->CreateShaderResourceView(mNormalDepth.Get(), &normalSrv, srvCpu);
}
