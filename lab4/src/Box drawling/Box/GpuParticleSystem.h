#pragma once

#include "../../Common/d3dUtil.h"

// Matches Particle in Shaders/particles.hlsl. Only the GPU changes these values.
struct GpuParticle
{
    DirectX::XMFLOAT3 Position;
    float Age;
    DirectX::XMFLOAT3 Velocity;
    float Lifetime;
    DirectX::XMFLOAT3 Color;
    float Radius;
};
static_assert(sizeof(GpuParticle) == 48, "Particle layout must match HLSL.");

class GpuParticleSystem
{
public:
    static constexpr UINT Capacity = 8192;

    void Initialize(ID3D12Device* device, ID3D12GraphicsCommandList* commands,
        DXGI_FORMAT depthFormat);
    void Reset(ID3D12GraphicsCommandList* commands);
    void Simulate(ID3D12GraphicsCommandList* commands, float deltaTime, UINT emitCount,
        const DirectX::XMFLOAT3& emitter, float floorY);
    // The caller binds the two G-buffer RTVs, depth buffer and viewport.
    void Draw(ID3D12GraphicsCommandList* commands,
        DirectX::FXMMATRIX view, DirectX::CXMMATRIX projection);

    // Used only by the explicit GPU test, never by the frame loop.
    // Readback layout: D3D12_DRAW_ARGUMENTS, followed by Capacity particles.
    static constexpr UINT64 ReadbackSize = sizeof(D3D12_DRAW_ARGUMENTS) +
        static_cast<UINT64>(Capacity) * sizeof(GpuParticle);
    void CopyReadback(ID3D12GraphicsCommandList* commands, ID3D12Resource* readback);

private:
    void CopyCount(ID3D12GraphicsCommandList* commands, UINT bufferIndex);
    void MakeWritable(ID3D12GraphicsCommandList* commands, UINT bufferIndex);
    D3D12_GPU_DESCRIPTOR_HANDLE Uav(UINT bufferIndex) const;

    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, 2> mParticles;
    std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, 2> mCounters;
    std::array<D3D12_RESOURCE_STATES, 2> mParticleStates =
        { D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS };
    Microsoft::WRL::ComPtr<ID3D12Resource> mCountSnapshot;
    Microsoft::WRL::ComPtr<ID3D12Resource> mDrawArguments;
    Microsoft::WRL::ComPtr<ID3D12Resource> mResetUpload;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mUavHeap;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> mComputeRoot;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> mGraphicsRoot;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mSimulatePso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mEmitPso;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mDrawPso;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> mDrawSignature;
    UINT mDescriptorSize = 0;
    UINT mActive = 0;
    UINT mSeed = 0;
};
