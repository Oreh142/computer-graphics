#pragma once

#include "../../Common/d3dUtil.h"

struct PostProcessSettings
{
    bool Bloom = true;
    bool Vignette = true;
    float Exposure = 2.0f;
    float BloomThreshold = 0.12f;
    float BloomStrength = 0.65f;
    float VignetteStrength = 0.45f;
};

class PostProcessing
{
public:
    static constexpr DXGI_FORMAT HdrFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
    static constexpr DXGI_FORMAT OutputFormat = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;

    bool IsInitialized() const { return mRoot != nullptr; }
    void Initialize(ID3D12Device* device);
    // Call after waiting for the GPU, including after replacing the G-buffer on resize.
    void Resize(ID3D12Device* device, UINT width, UINT height,
        ID3D12Resource* albedo, ID3D12Resource* normal, ID3D12Resource* depth);
    void BeginScene(ID3D12GraphicsCommandList* commands);
    void Apply(ID3D12GraphicsCommandList* commands, D3D12_CPU_DESCRIPTOR_HANDLE output,
        const PostProcessSettings& settings, UINT debugView);

private:
    D3D12_CPU_DESCRIPTOR_HANDLE Rtv(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE Srv(UINT index) const;
    void DrawQuad(ID3D12GraphicsCommandList* commands, ID3D12PipelineState* pso,
        D3D12_CPU_DESCRIPTOR_HANDLE output, UINT width, UINT height);

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRoot;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mExtractPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mBlurPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mCompositePso;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mRtvHeap;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mSrvHeap;
    // HDR scene, bloom A, bloom B. All return to PIXEL_SHADER_RESOURCE after Apply.
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, 3> mTargets;
    UINT mWidth = 1, mHeight = 1, mBloomWidth = 1, mBloomHeight = 1;
    UINT mRtvSize = 0, mSrvSize = 0;
};
