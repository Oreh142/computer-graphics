#include "../GpuParticleSystem.h"
#include "../GBuffer.h"
#include "TestGpu.h"
using namespace GpuTests;
#include <d3d12sdklayers.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    std::vector<GpuParticle> ReadParticles(TestGpu& gpu, GpuParticleSystem& particles)
    {
        auto readback = gpu.Readback(GpuParticleSystem::ReadbackSize);
        particles.CopyReadback(gpu.Commands.Get(), readback.Get());
        gpu.Submit();
        void* mapped = nullptr;
        const D3D12_RANGE range = { 0, static_cast<SIZE_T>(GpuParticleSystem::ReadbackSize) };
        ThrowIfFailed(readback->Map(0, &range, &mapped));
        D3D12_DRAW_ARGUMENTS args;
        memcpy(&args, mapped, sizeof(args));
        Require(args.VertexCountPerInstance <= GpuParticleSystem::Capacity, "Counter overflow");
        Require(args.InstanceCount == 1 && args.StartVertexLocation == 0 && args.StartInstanceLocation == 0,
            "Invalid indirect draw arguments");
        std::vector<GpuParticle> result(args.VertexCountPerInstance);
        if (!result.empty())
            memcpy(result.data(), static_cast<char*>(mapped) + sizeof(args), result.size() * sizeof(GpuParticle));
        const D3D12_RANGE noWrite = { 0, 0 };
        readback->Unmap(0, &noWrite);
        return result;
    }

    void TestBillboards(TestGpu& gpu, GpuParticleSystem& particles)
    {
        constexpr UINT size = 256;
        GBuffer buffers;
        buffers.Build(gpu.Device.Get(), size, size);
        const auto depthDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D24_UNORM_S8_UINT,
            size, size, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        std::array<ComPtr<ID3D12Resource>, 2> depth;
        D3D12_DESCRIPTOR_HEAP_DESC dsvDesc = {};
        dsvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvDesc.NumDescriptors = 2;
        ComPtr<ID3D12DescriptorHeap> dsvHeap;
        ThrowIfFailed(gpu.Device->CreateDescriptorHeap(&dsvDesc, IID_PPV_ARGS(&dsvHeap)));
        const UINT dsvSize = gpu.Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        for (UINT i = 0; i < 2; ++i)
        {
            D3D12_CLEAR_VALUE clear = {};
            clear.Format = depthDesc.Format;
            clear.DepthStencil.Depth = i == 0 ? 1.0f : 0.0f;
            ThrowIfFailed(gpu.Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                &depthDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear, IID_PPV_ARGS(&depth[i])));
            auto handle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<SIZE_T>(i) * dsvSize;
            gpu.Device->CreateDepthStencilView(depth[i].Get(), nullptr, handle);
        }
        auto albedoBarrier = CD3DX12_RESOURCE_BARRIER::Transition(buffers.AlbedoResource(),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        auto normalBarrier = CD3DX12_RESOURCE_BARRIER::Transition(buffers.NormalResource(),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        gpu.Commands->ResourceBarrier(1, &albedoBarrier);
        gpu.Commands->ResourceBarrier(1, &normalBarrier);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvs[] = { buffers.AlbedoRtv(), buffers.NormalRtv() };
        const D3D12_VIEWPORT viewport = { 0, 0, float(size), float(size), 0, 1 };
        const D3D12_RECT scissor = { 0, 0, size, size };
        const auto imageDesc = buffers.AlbedoResource()->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        UINT64 bytes = 0;
        gpu.Device->GetCopyableFootprints(&imageDesc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        auto imageReadback = gpu.Readback(bytes);

        // A second camera checks that GS billboards follow camera orientation.
        for (UINT pass = 0; pass < 3; ++pass)
        {
            auto* commands = gpu.Commands.Get();
            auto dsv = dsvHeap->GetCPUDescriptorHandleForHeapStart();
            if (pass == 2) dsv.ptr += dsvSize;
            commands->RSSetViewports(1, &viewport);
            commands->RSSetScissorRects(1, &scissor);
            const float black[4] = { 0, 0, 0, 1 };
            const float normalClear[4] = { 0.5f, 0.5f, 0, 1 };
            commands->ClearRenderTargetView(rtvs[0], black, 0, nullptr);
            commands->ClearRenderTargetView(rtvs[1], normalClear, 0, nullptr);
            // Depth zero covers the whole viewport with a nearer occluder.
            commands->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,
                pass == 2 ? 0.0f : 1.0f, 0, 0, nullptr);
            commands->OMSetRenderTargets(2, rtvs, FALSE, &dsv);
            const XMVECTOR eye = pass == 1 ? XMVectorSet(5, 3, 0, 1) : XMVectorSet(0, 3, -5, 1);
            const auto view = XMMatrixLookAtLH(eye, XMVectorSet(0, 2.5f, 0, 1), XMVectorSet(0, 1, 0, 0));
            particles.Draw(commands, view, XMMatrixPerspectiveFovLH(XM_PIDIV4, 1.0f, 0.1f, 100.0f));
            auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(buffers.AlbedoResource(),
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->ResourceBarrier(1, &toCopy);
            const CD3DX12_TEXTURE_COPY_LOCATION source(buffers.AlbedoResource(), 0);
            const CD3DX12_TEXTURE_COPY_LOCATION target(imageReadback.Get(), footprint);
            commands->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
            auto fromCopy = CD3DX12_RESOURCE_BARRIER::Transition(buffers.AlbedoResource(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            commands->ResourceBarrier(1, &fromCopy);
            gpu.Submit();
            void* mapped = nullptr;
            const D3D12_RANGE range = { 0, static_cast<SIZE_T>(bytes) };
            ThrowIfFailed(imageReadback->Map(0, &range, &mapped));
            UINT visiblePixels = 0;
            std::ofstream preview;
            if (pass == 0)
            {
                preview.open("x64/ParticleTests/particles.ppm", std::ios::binary);
                preview << "P6\n" << size << ' ' << size << "\n255\n";
            }
            for (UINT y = 0; y < size; ++y)
            {
                auto* row = static_cast<unsigned char*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch;
                for (UINT x = 0; x < size; ++x)
                {
                    const auto* half = reinterpret_cast<const uint16_t*>(row + x * 8);
                    unsigned char pixel[4];
                    for (UINT channel = 0; channel < 4; ++channel)
                        pixel[channel] = static_cast<unsigned char>(255.0f * (std::min)(1.0f,
                            DirectX::PackedVector::XMConvertHalfToFloat(half[channel])) + 0.5f);
                    if (pixel[0] || pixel[1] || pixel[2]) ++visiblePixels;
                    Require(pixel[3] == 255, "Particles must be opaque");
                    if (pass == 0) preview.write(reinterpret_cast<const char*>(pixel), 3);
                }
            }
            const D3D12_RANGE noWrite = { 0, 0 };
            imageReadback->Unmap(0, &noWrite);
            Require(pass == 2 ? visiblePixels == 0 : visiblePixels > 50,
                "Billboard rendering or depth occlusion failed");
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        const bool hardware = argc > 1 && std::string(argv[1]) == "--hardware";
        TestGpu gpu(hardware);
        GpuParticleSystem particles;
        particles.Initialize(gpu.Device.Get(), gpu.Commands.Get(), DXGI_FORMAT_D24_UNORM_S8_UINT);
        Require(ReadParticles(gpu, particles).empty(), "Initialization must produce zero particles");

        const XMFLOAT3 emitter(0.0f, 1.0f, 0.0f);
        particles.Simulate(gpu.Commands.Get(), 0.0f, 257, emitter, 0.0f);
        auto born = ReadParticles(gpu, particles);
        Require(born.size() == 257, "Partial thread group emission failed");
        for (const auto& p : born)
            Require(p.Age == 0 && p.Lifetime >= 2 && p.Radius > 0, "Invalid particle initialization");

        particles.Simulate(gpu.Commands.Get(), 1.0f / 60.0f, 0, emitter, 0.0f);
        auto moved = ReadParticles(gpu, particles);
        Require(moved.size() == born.size(), "Live particles were lost");
        // Append/Consume order is deliberately unspecified. Match by unchanged attributes.
        auto less = [](const GpuParticle& a, const GpuParticle& b) { return a.Radius < b.Radius; };
        std::sort(born.begin(), born.end(), less);
        std::sort(moved.begin(), moved.end(), less);
        for (size_t i = 0; i < born.size(); ++i)
        {
            const float expectedY = born[i].Position.y + born[i].Velocity.y / 60.0f - 2.0f / 3600.0f;
            Require(std::abs(moved[i].Position.y - expectedY) < 0.0001f &&
                std::abs(moved[i].Age - 1.0f / 60.0f) < 0.00001f, "GPU integration failed");
        }

        for (UINT i = 0; i < 45; ++i)
            particles.Simulate(gpu.Commands.Get(), 1.0f / 60.0f, 20, emitter, 0.0f);
        TestBillboards(gpu, particles);

        particles.Reset(gpu.Commands.Get());
        Require(ReadParticles(gpu, particles).empty(), "Reset failed");
        particles.Simulate(gpu.Commands.Get(), 0.0f, UINT_MAX, emitter, 0.0f);
        Require(ReadParticles(gpu, particles).size() == GpuParticleSystem::Capacity, "Capacity fill failed");
        for (UINT i = 0; i < 3; ++i)
            particles.Simulate(gpu.Commands.Get(), 0.0f, 256, emitter, 0.0f);
        Require(ReadParticles(gpu, particles).size() == GpuParticleSystem::Capacity, "Capacity overflow or ping-pong loss");
        for (UINT i = 0; i < 130; ++i)
            particles.Simulate(gpu.Commands.Get(), 1.0f / 30.0f, 0, emitter, -100.0f);
        Require(ReadParticles(gpu, particles).empty(), "Expired particles were not removed");
        particles.Simulate(gpu.Commands.Get(), 0.0f, 17, emitter, 0.0f);
        Require(ReadParticles(gpu, particles).size() == 17, "Emission after an empty frame failed");
        particles.Simulate(gpu.Commands.Get(), 0.0f, 0, emitter, 100.0f);
        Require(ReadParticles(gpu, particles).empty(), "Floor removal failed");
        std::cout << "PASS (" << (hardware ? "hardware" : "WARP")
            << "): emission, GPU motion, lifetime, floor, capacity, ping-pong, reset, indirect draw, "
            << "billboards from two cameras, opacity, depth occlusion; D3D12 GPU validation clean.\n";
        return 0;
    }
    catch (const DxException& error)
    {
        std::wcerr << error.ToString() << L'\n';
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
    }
    return 1;
}
