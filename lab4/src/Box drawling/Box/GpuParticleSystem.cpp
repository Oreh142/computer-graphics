#include "GpuParticleSystem.h"
#include "GBuffer.h"

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    DXGI_FORMAT DepthSrvFormat(DXGI_FORMAT depthFormat)
    {
        switch (depthFormat)
        {
        case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_UNORM;
        case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT;
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
        default: throw std::runtime_error("Unsupported particle collision depth format");
        }
    }

    void Transition(ID3D12GraphicsCommandList* commands, ID3D12Resource* resource,
        D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        if (before == after)
            return;
        auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(resource, before, after);
        commands->ResourceBarrier(1, &barrier);
    }

    ComPtr<ID3D12Resource> Buffer(ID3D12Device* device, ID3D12GraphicsCommandList* commands, UINT64 bytes,
        D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags)
    {
        ComPtr<ID3D12Resource> result;
        const auto heap = CD3DX12_HEAP_PROPERTIES(heapType);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);
        const auto initialState = heapType == D3D12_HEAP_TYPE_DEFAULT ? D3D12_RESOURCE_STATE_COMMON : state;
        ThrowIfFailed(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &desc, initialState, nullptr, IID_PPV_ARGS(&result)));
        Transition(commands, result.Get(), initialState, state);
        return result;
    }

    ComPtr<ID3D12RootSignature> RootSignature(ID3D12Device* device,
        const CD3DX12_ROOT_SIGNATURE_DESC& desc)
    {
        ComPtr<ID3DBlob> blob, errors;
        ThrowIfFailed(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
            &blob, &errors));
        ComPtr<ID3D12RootSignature> root;
        ThrowIfFailed(device->CreateRootSignature(0, blob->GetBufferPointer(),
            blob->GetBufferSize(), IID_PPV_ARGS(&root)));
        return root;
    }
}

void GpuParticleSystem::Initialize(ID3D12Device* device,
    ID3D12GraphicsCommandList* commands, DXGI_FORMAT depthFormat)
{
    mDepthSrvFormat = DepthSrvFormat(depthFormat);
    mResetUpload = Buffer(device, commands, sizeof(D3D12_DRAW_ARGUMENTS), D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);
    const D3D12_DRAW_ARGUMENTS emptyDraw = { 0, 1, 0, 0 };
    void* mapped = nullptr;
    const D3D12_RANGE noRead = { 0, 0 };
    ThrowIfFailed(mResetUpload->Map(0, &noRead, &mapped));
    memcpy(mapped, &emptyDraw, sizeof(emptyDraw));
    mResetUpload->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // Two append/consume UAVs and one SRV for screen-space depth collision.
    heapDesc.NumDescriptors = 3;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&mUavHeap)));
    mDescriptorSize = device->GetDescriptorHandleIncrementSize(heapDesc.Type);

    D3D12_SHADER_RESOURCE_VIEW_DESC nullDepthSrv = {};
    nullDepthSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    nullDepthSrv.Format = mDepthSrvFormat;
    nullDepthSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    nullDepthSrv.Texture2D.MipLevels = 1;
    auto depthHandle = mUavHeap->GetCPUDescriptorHandleForHeapStart();
    depthHandle.ptr += static_cast<SIZE_T>(2) * mDescriptorSize;
    device->CreateShaderResourceView(nullptr, &nullDepthSrv, depthHandle);

    for (UINT i = 0; i < 2; ++i)
    {
        mParticles[i] = Buffer(device, commands, static_cast<UINT64>(Capacity) * sizeof(GpuParticle),
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        // Each UAV owns a separate counter, at aligned offset zero.
        mCounters[i] = Buffer(device, commands, sizeof(UINT), D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        commands->CopyBufferRegion(mCounters[i].Get(), 0, mResetUpload.Get(), 0, sizeof(UINT));
        Transition(commands, mCounters[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = Capacity;
        uav.Buffer.StructureByteStride = sizeof(GpuParticle);
        auto handle = mUavHeap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += static_cast<SIZE_T>(i) * mDescriptorSize;
        device->CreateUnorderedAccessView(mParticles[i].Get(), mCounters[i].Get(), &uav, handle);
    }
    mCountSnapshot = Buffer(device, commands, sizeof(UINT), D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_FLAG_NONE);
    mDrawArguments = Buffer(device, commands, sizeof(emptyDraw), D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);
    commands->CopyBufferRegion(mDrawArguments.Get(), 0, mResetUpload.Get(), 0, sizeof(emptyDraw));
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);

    CD3DX12_DESCRIPTOR_RANGE inputRange, outputRange, depthRange;
    inputRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0);
    outputRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 1);
    depthRange.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2);
    CD3DX12_ROOT_PARAMETER computeParams[5];
    computeParams[0].InitAsConstants(52, 0);
    computeParams[1].InitAsDescriptorTable(1, &inputRange);
    computeParams[2].InitAsDescriptorTable(1, &outputRange);
    computeParams[3].InitAsShaderResourceView(1);
    computeParams[4].InitAsDescriptorTable(1, &depthRange);
    mComputeRoot = RootSignature(device, CD3DX12_ROOT_SIGNATURE_DESC(5, computeParams));

    CD3DX12_ROOT_PARAMETER graphicsParams[2];
    graphicsParams[0].InitAsConstants(24, 1);
    graphicsParams[1].InitAsShaderResourceView(0, 0, D3D12_SHADER_VISIBILITY_VERTEX);
    mGraphicsRoot = RootSignature(device, CD3DX12_ROOT_SIGNATURE_DESC(2, graphicsParams));

    auto simulate = d3dUtil::CompileShader(L"Shaders\\particles.hlsl", nullptr, "SimulateCS", "cs_5_0");
    auto emit = d3dUtil::CompileShader(L"Shaders\\particles.hlsl", nullptr, "EmitCS", "cs_5_0");
    D3D12_COMPUTE_PIPELINE_STATE_DESC compute = {};
    compute.pRootSignature = mComputeRoot.Get();
    compute.CS = { simulate->GetBufferPointer(), simulate->GetBufferSize() };
    ThrowIfFailed(device->CreateComputePipelineState(&compute, IID_PPV_ARGS(&mSimulatePso)));
    compute.CS = { emit->GetBufferPointer(), emit->GetBufferSize() };
    ThrowIfFailed(device->CreateComputePipelineState(&compute, IID_PPV_ARGS(&mEmitPso)));

    auto vs = d3dUtil::CompileShader(L"Shaders\\particles.hlsl", nullptr, "ParticleVS", "vs_5_0");
    auto gs = d3dUtil::CompileShader(L"Shaders\\particles.hlsl", nullptr, "ParticleGS", "gs_5_0");
    auto ps = d3dUtil::CompileShader(L"Shaders\\particles.hlsl", nullptr, "ParticlePS", "ps_5_0");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC graphics = {};
    graphics.pRootSignature = mGraphicsRoot.Get();
    graphics.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
    graphics.GS = { gs->GetBufferPointer(), gs->GetBufferSize() };
    graphics.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
    graphics.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    graphics.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    graphics.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    graphics.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    graphics.SampleMask = UINT_MAX;
    graphics.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    graphics.NumRenderTargets = 2;
    graphics.RTVFormats[0] = GBuffer::AlbedoFormat;
    graphics.RTVFormats[1] = DXGI_FORMAT_R16G16_FLOAT;
    graphics.DSVFormat = depthFormat;
    graphics.SampleDesc.Count = 1;
    ThrowIfFailed(device->CreateGraphicsPipelineState(&graphics, IID_PPV_ARGS(&mDrawPso)));

    D3D12_INDIRECT_ARGUMENT_DESC draw = {};
    draw.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    D3D12_COMMAND_SIGNATURE_DESC signature = {};
    signature.ByteStride = sizeof(D3D12_DRAW_ARGUMENTS);
    signature.NumArgumentDescs = 1;
    signature.pArgumentDescs = &draw;
    ThrowIfFailed(device->CreateCommandSignature(&signature, nullptr, IID_PPV_ARGS(&mDrawSignature)));
}

void GpuParticleSystem::SetCollisionDepth(ID3D12Device* device, ID3D12Resource* depthBuffer)
{
    assert(device && depthBuffer && mUavHeap);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = mDepthSrvFormat;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Texture2D.MostDetailedMip = 0;
    srv.Texture2D.MipLevels = 1;
    auto handle = mUavHeap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(2) * mDescriptorSize;
    device->CreateShaderResourceView(depthBuffer, &srv, handle);
}

D3D12_GPU_DESCRIPTOR_HANDLE GpuParticleSystem::Uav(UINT bufferIndex) const
{
    auto handle = mUavHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(bufferIndex) * mDescriptorSize;
    return handle;
}

D3D12_GPU_DESCRIPTOR_HANDLE GpuParticleSystem::DepthSrv() const
{
    auto handle = mUavHeap->GetGPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<UINT64>(2) * mDescriptorSize;
    return handle;
}

void GpuParticleSystem::MakeWritable(ID3D12GraphicsCommandList* commands, UINT bufferIndex)
{
    Transition(commands, mParticles[bufferIndex].Get(), mParticleStates[bufferIndex],
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    mParticleStates[bufferIndex] = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
}

void GpuParticleSystem::CopyCount(ID3D12GraphicsCommandList* commands, UINT bufferIndex)
{
    Transition(commands, mCounters[bufferIndex].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(commands, mCountSnapshot.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST);
    commands->CopyBufferRegion(mCountSnapshot.Get(), 0, mCounters[bufferIndex].Get(), 0, sizeof(UINT));
    Transition(commands, mCountSnapshot.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Transition(commands, mCounters[bufferIndex].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void GpuParticleSystem::Reset(ID3D12GraphicsCommandList* commands)
{
    for (UINT i = 0; i < 2; ++i)
    {
        MakeWritable(commands, i);
        Transition(commands, mCounters[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST);
        commands->CopyBufferRegion(mCounters[i].Get(), 0, mResetUpload.Get(), 0, sizeof(UINT));
        Transition(commands, mCounters[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COPY_DEST);
    commands->CopyBufferRegion(mDrawArguments.Get(), 0, mResetUpload.Get(), 0, sizeof(D3D12_DRAW_ARGUMENTS));
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    mActive = 0;
    mSeed = 0;
}

void GpuParticleSystem::Simulate(ID3D12GraphicsCommandList* commands, float deltaTime,
    UINT emitCount, const XMFLOAT3& emitter, float floorY)
{
    const XMMATRIX identity = XMMatrixIdentity();
    SimulateInternal(commands, deltaTime, emitCount, emitter, floorY, false,
        identity, identity, XMFLOAT3(0.0f, 0.0f, 0.0f), 1, 1);
}

void GpuParticleSystem::SimulateWithDepth(ID3D12GraphicsCommandList* commands, float deltaTime,
    UINT emitCount, const XMFLOAT3& emitter, float floorY, ID3D12Resource* depthBuffer,
    FXMMATRIX view, CXMMATRIX projection, const XMFLOAT3& cameraPosition,
    UINT depthWidth, UINT depthHeight)
{
    assert(depthBuffer && depthWidth > 0 && depthHeight > 0);
    Transition(commands, depthBuffer, D3D12_RESOURCE_STATE_DEPTH_WRITE,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    SimulateInternal(commands, deltaTime, emitCount, emitter, floorY, true,
        view, projection, cameraPosition, depthWidth, depthHeight);
    Transition(commands, depthBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_DEPTH_WRITE);
}

void GpuParticleSystem::SimulateInternal(ID3D12GraphicsCommandList* commands, float deltaTime,
    UINT emitCount, const XMFLOAT3& emitter, float floorY, bool depthCollision,
    FXMMATRIX view, CXMMATRIX projection, const XMFLOAT3& cameraPosition,
    UINT depthWidth, UINT depthHeight)
{
    const UINT output = 1 - mActive;
    MakeWritable(commands, mActive);
    MakeWritable(commands, output);
    Transition(commands, mCounters[output].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_DEST);
    commands->CopyBufferRegion(mCounters[output].Get(), 0, mResetUpload.Get(), 0, sizeof(UINT));
    Transition(commands, mCounters[output].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    CopyCount(commands, mActive);

    struct SimulationConstants
    {
        float DeltaTime;
        UINT EmitCount, Seed, MaxParticles;
        XMFLOAT3 Emitter;
        float FloorY;
        XMFLOAT3 Acceleration;
        float Padding;
        XMFLOAT4X4 ViewProjection;
        XMFLOAT4X4 InverseViewProjection;
        XMFLOAT3 CameraPosition;
        float CollisionThickness;
        XMUINT2 DepthDimensions;
        UINT CollisionEnabled;
        float Restitution;
    };
    SimulationConstants constants = {};
    constants.DeltaTime = (std::max)(0.0f, (std::min)(deltaTime, 1.0f / 30.0f));
    constants.EmitCount = (std::min)(emitCount, Capacity);
    constants.Seed = ++mSeed;
    constants.MaxParticles = Capacity;
    constants.Emitter = emitter;
    constants.FloorY = floorY;
    constants.Acceleration = XMFLOAT3(0.25f, -4.0f, 0.0f);
    const XMMATRIX viewProjection = view * projection;
    XMStoreFloat4x4(&constants.ViewProjection, XMMatrixTranspose(viewProjection));
    XMStoreFloat4x4(&constants.InverseViewProjection,
        XMMatrixTranspose(XMMatrixInverse(nullptr, viewProjection)));
    constants.CameraPosition = cameraPosition;
    constants.CollisionThickness = 0.01f;
    constants.DepthDimensions = XMUINT2(depthWidth, depthHeight);
    constants.CollisionEnabled = depthCollision ? 1u : 0u;
    constants.Restitution = 0.55f;
    static_assert(sizeof(constants) == 208, "Simulation constants must match HLSL.");
    ID3D12DescriptorHeap* heaps[] = { mUavHeap.Get() };
    commands->SetDescriptorHeaps(1, heaps);
    commands->SetComputeRootSignature(mComputeRoot.Get());
    commands->SetComputeRoot32BitConstants(0, 52, &constants, 0);
    commands->SetComputeRootDescriptorTable(1, Uav(mActive));
    commands->SetComputeRootDescriptorTable(2, Uav(output));
    commands->SetComputeRootShaderResourceView(3, mCountSnapshot->GetGPUVirtualAddress());
    commands->SetComputeRootDescriptorTable(4, DepthSrv());
    commands->SetPipelineState(mSimulatePso.Get());
    commands->Dispatch((Capacity + 255) / 256, 1, 1);
    auto uavBarrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    commands->ResourceBarrier(1, &uavBarrier);

    if (constants.EmitCount > 0)
    {
        CopyCount(commands, output);
        commands->SetPipelineState(mEmitPso.Get());
        commands->Dispatch((constants.EmitCount + 255) / 256, 1, 1);
        commands->ResourceBarrier(1, &uavBarrier);
    }

    // The GPU counter becomes VertexCountPerInstance; the CPU never reads it.
    Transition(commands, mCounters[output].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COPY_DEST);
    commands->CopyBufferRegion(mDrawArguments.Get(), 0, mCounters[output].Get(), 0, sizeof(UINT));
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
    Transition(commands, mCounters[output].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    mActive = output;
}

void GpuParticleSystem::Draw(ID3D12GraphicsCommandList* commands,
    FXMMATRIX view, CXMMATRIX projection)
{
    Transition(commands, mParticles[mActive].Get(), mParticleStates[mActive],
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    mParticleStates[mActive] = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    struct CameraConstants
    {
        XMFLOAT4X4 ViewProjection;
        XMFLOAT3 Right;
        float Padding0 = 0.0f;
        XMFLOAT3 Up;
        float Padding1 = 0.0f;
    } camera;
    static_assert(sizeof(camera) == 96, "Camera constants must match HLSL.");
    XMStoreFloat4x4(&camera.ViewProjection, XMMatrixTranspose(view * projection));
    const XMMATRIX invView = XMMatrixInverse(nullptr, view);
    XMStoreFloat3(&camera.Right, invView.r[0]);
    XMStoreFloat3(&camera.Up, invView.r[1]);
    commands->SetGraphicsRootSignature(mGraphicsRoot.Get());
    commands->SetGraphicsRoot32BitConstants(0, 24, &camera, 0);
    commands->SetGraphicsRootShaderResourceView(1, mParticles[mActive]->GetGPUVirtualAddress());
    commands->SetPipelineState(mDrawPso.Get());
    commands->IASetVertexBuffers(0, 0, nullptr);
    commands->IASetIndexBuffer(nullptr);
    commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
    commands->ExecuteIndirect(mDrawSignature.Get(), 1, mDrawArguments.Get(), 0, nullptr, 0);
}

void GpuParticleSystem::CopyReadback(ID3D12GraphicsCommandList* commands, ID3D12Resource* readback)
{
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    Transition(commands, mParticles[mActive].Get(), mParticleStates[mActive], D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands->CopyBufferRegion(readback, 0, mDrawArguments.Get(), 0, sizeof(D3D12_DRAW_ARGUMENTS));
    commands->CopyBufferRegion(readback, sizeof(D3D12_DRAW_ARGUMENTS), mParticles[mActive].Get(),
        0, static_cast<UINT64>(Capacity) * sizeof(GpuParticle));
    Transition(commands, mParticles[mActive].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, mParticleStates[mActive]);
    Transition(commands, mDrawArguments.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
}
