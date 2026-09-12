#include "PostProcessing.h"

using Microsoft::WRL::ComPtr;

namespace
{
    void ChangeState(ID3D12GraphicsCommandList* commands, ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, before, after);
        commands->ResourceBarrier(1, &barrier);
    }

    struct Constants
    {
        DirectX::XMFLOAT2 InvSourceSize;
        DirectX::XMFLOAT2 BlurDirection = { 0.0f, 0.0f };
        float Exposure, BloomThreshold, BloomStrength, VignetteStrength;
        UINT BloomEnabled, VignetteEnabled, DebugView;
        float Aspect;
    };
    static_assert(sizeof(Constants) == 48, "Post-process constants must match HLSL.");
}

void PostProcessing::Initialize(ID3D12Device* device)
{
    CD3DX12_DESCRIPTOR_RANGE ranges[3];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    ranges[1].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 1);
    ranges[2].Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 4);
    CD3DX12_ROOT_PARAMETER params[4];
    for (UINT i = 0; i < 3; ++i)
        params[i].InitAsDescriptorTable(1, &ranges[i], D3D12_SHADER_VISIBILITY_PIXEL);
    params[3].InitAsConstants(12, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL);
    const CD3DX12_STATIC_SAMPLER_DESC sampler(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP);
    const CD3DX12_ROOT_SIGNATURE_DESC rootDesc(4, params, 1, &sampler);
    ComPtr<ID3DBlob> serialized, errors;
    ThrowIfFailed(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &errors));
    ThrowIfFailed(device->CreateRootSignature(0, serialized->GetBufferPointer(), serialized->GetBufferSize(), IID_PPV_ARGS(&mRoot)));

    auto vs = d3dUtil::CompileShader(L"Shaders\\postprocess.hlsl", nullptr, "FullscreenVS", "vs_5_0");
    auto makePso = [&](const char* entry, DXGI_FORMAT format, ComPtr<ID3D12PipelineState>& result)
    {
        auto ps = d3dUtil::CompileShader(L"Shaders\\postprocess.hlsl", nullptr, entry, "ps_5_0");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc = {};
        desc.pRootSignature = mRoot.Get();
        desc.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        desc.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        desc.DepthStencilState.DepthEnable = FALSE;
        desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        desc.SampleMask = UINT_MAX;
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets = 1;
        desc.RTVFormats[0] = format;
        desc.SampleDesc.Count = 1;
        ThrowIfFailed(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&result)));
    };
    makePso("BloomExtractPS", HdrFormat, mExtractPso);
    makePso("GaussianBlurPS", HdrFormat, mBlurPso);
    makePso("PostProcessPS", OutputFormat, mCompositePso);

    D3D12_DESCRIPTOR_HEAP_DESC heap = {};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    heap.NumDescriptors = 3;
    ThrowIfFailed(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&mRtvHeap)));
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = 6;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&mSrvHeap)));
    mRtvSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    mSrvSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

D3D12_CPU_DESCRIPTOR_HANDLE PostProcessing::Rtv(UINT index) const
{
    auto handle = mRtvHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * mRtvSize;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE PostProcessing::Srv(UINT index) const
{
    auto handle = mSrvHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(index) * mSrvSize;
    return handle;
}

void PostProcessing::Resize(ID3D12Device* device, UINT width, UINT height,
    ID3D12Resource* albedo, ID3D12Resource* normal, ID3D12Resource* depth)
{
    if (width == 0 || height == 0) return;
    mWidth = width;
    mHeight = height;
    mBloomWidth = (width + 1) / 2;
    mBloomHeight = (height + 1) / 2;
    auto makeSrv = [&](ID3D12Resource* resource, DXGI_FORMAT format, UINT index)
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC view = {};
        view.Format = format;
        view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        view.Texture2D.MipLevels = 1;
        auto cpu = mSrvHeap->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += static_cast<SIZE_T>(index) * mSrvSize;
        device->CreateShaderResourceView(resource, &view, cpu);
    };
    for (UINT i = 0; i < 3; ++i)
    {
        mTargets[i].Reset();
        const auto desc = CD3DX12_RESOURCE_DESC::Tex2D(HdrFormat,
            i == 0 ? mWidth : mBloomWidth, i == 0 ? mHeight : mBloomHeight,
            1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        D3D12_CLEAR_VALUE clear = {};
        clear.Format = HdrFormat;
        clear.Color[3] = 1.0f;
        ThrowIfFailed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, &clear, IID_PPV_ARGS(&mTargets[i])));
        device->CreateRenderTargetView(mTargets[i].Get(), nullptr, Rtv(i));
        makeSrv(mTargets[i].Get(), HdrFormat, i == 0 ? 0 : i + 3);
    }
    makeSrv(albedo, albedo->GetDesc().Format, 1);
    makeSrv(normal, normal->GetDesc().Format, 2);
    makeSrv(depth, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, 3);
}

void PostProcessing::BeginScene(ID3D12GraphicsCommandList* commands)
{
    ChangeState(commands, mTargets[0].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    const auto target = Rtv(0);
    const float black[4] = { 0, 0, 0, 1 };
    commands->ClearRenderTargetView(target, black, 0, nullptr);
    commands->OMSetRenderTargets(1, &target, FALSE, nullptr);
}

void PostProcessing::DrawQuad(ID3D12GraphicsCommandList* commands, ID3D12PipelineState* pso,
    D3D12_CPU_DESCRIPTOR_HANDLE output, UINT width, UINT height)
{
    const D3D12_VIEWPORT viewport = { 0, 0, float(width), float(height), 0, 1 };
    const D3D12_RECT scissor = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
    commands->RSSetViewports(1, &viewport);
    commands->RSSetScissorRects(1, &scissor);
    commands->OMSetRenderTargets(1, &output, FALSE, nullptr);
    commands->SetPipelineState(pso);
    commands->IASetVertexBuffers(0, 0, nullptr);
    commands->IASetIndexBuffer(nullptr);
    commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    commands->DrawInstanced(4, 1, 0, 0);
}

void PostProcessing::Apply(ID3D12GraphicsCommandList* commands, D3D12_CPU_DESCRIPTOR_HANDLE output,
    const PostProcessSettings& settings, UINT debugView)
{
    ChangeState(commands, mTargets[0].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* heaps[] = { mSrvHeap.Get() };
    commands->SetDescriptorHeaps(1, heaps);
    commands->SetGraphicsRootSignature(mRoot.Get());
    commands->SetGraphicsRootDescriptorTable(1, Srv(1));
    // Bind initialized HDR data when bloom is skipped; never sample an unwritten target.
    commands->SetGraphicsRootDescriptorTable(2, Srv(0));
    Constants constants = {};
    constants.InvSourceSize = { 1.0f / mWidth, 1.0f / mHeight };
    constants.Exposure = settings.Exposure;
    constants.BloomThreshold = settings.BloomThreshold;
    constants.BloomStrength = settings.BloomStrength;
    constants.VignetteStrength = settings.VignetteStrength;
    constants.BloomEnabled = settings.Bloom && (debugView == 0 || debugView == 4);
    constants.VignetteEnabled = settings.Vignette;
    constants.DebugView = debugView;
    constants.Aspect = float(mWidth) / mHeight;
    auto bind = [&](UINT source)
    {
        commands->SetGraphicsRootDescriptorTable(0, Srv(source));
        commands->SetGraphicsRoot32BitConstants(3, 12, &constants, 0);
    };

    if (constants.BloomEnabled)
    {
        ChangeState(commands, mTargets[1].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        bind(0);
        DrawQuad(commands, mExtractPso.Get(), Rtv(1), mBloomWidth, mBloomHeight);
        ChangeState(commands, mTargets[1].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        constants.InvSourceSize = { 1.0f / mBloomWidth, 1.0f / mBloomHeight };
        // Two horizontal/vertical pairs give a wider glow at half resolution.
        for (UINT i = 0; i < 4; ++i)
        {
            const UINT destination = i % 2 == 0 ? 2 : 1;
            const UINT source = i % 2 == 0 ? 4 : 5;
            constants.BlurDirection = i % 2 == 0 ? DirectX::XMFLOAT2(1, 0) : DirectX::XMFLOAT2(0, 1);
            ChangeState(commands, mTargets[destination].Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            bind(source);
            DrawQuad(commands, mBlurPso.Get(), Rtv(destination), mBloomWidth, mBloomHeight);
            ChangeState(commands, mTargets[destination].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        commands->SetGraphicsRootDescriptorTable(2, Srv(4));
    }
    constants.InvSourceSize = { 1.0f / mWidth, 1.0f / mHeight };
    bind(0);
    DrawQuad(commands, mCompositePso.Get(), output, mWidth, mHeight);
}
