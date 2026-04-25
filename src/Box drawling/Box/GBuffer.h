#pragma once

#include "../../Common/d3dUtil.h"

class GBuffer
{
public:
    bool IsInitialized() const;

    void Build(ID3D12Device* device, UINT width, UINT height);
    void Resize(ID3D12Device* device, UINT width, UINT height);

    ID3D12DescriptorHeap* SrvHeap() const;
    ID3D12Resource* AlbedoResource() const;
    ID3D12Resource* NormalResource() const;
    D3D12_CPU_DESCRIPTOR_HANDLE AlbedoRtv() const;
    D3D12_CPU_DESCRIPTOR_HANDLE NormalRtv() const;

private:
    void CreateTextures(ID3D12Device* device);
    void CreateViews(ID3D12Device* device);

private:
    UINT mWidth = 1;
    UINT mHeight = 1;
    Microsoft::WRL::ComPtr<ID3D12Resource> mAlbedo;
    Microsoft::WRL::ComPtr<ID3D12Resource> mNormal;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRtvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mSrvHeap;
    D3D12_CPU_DESCRIPTOR_HANDLE mAlbedoRtv = {};
    D3D12_CPU_DESCRIPTOR_HANDLE mNormalRtv = {};
};
