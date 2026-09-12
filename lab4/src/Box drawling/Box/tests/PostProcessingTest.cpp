#include "../PostProcessing.h"
#include "../GBuffer.h"
#include "TestGpu.h"
#include <cmath>

using namespace GpuTests;
using Microsoft::WRL::ComPtr;

namespace
{
    unsigned char Encode(float linear)
    {
        const float srgb = linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        return static_cast<unsigned char>(srgb * 255.0f + 0.5f);
    }

    void Near(int actual, int expected, const char* message)
    {
        Require(std::abs(actual - expected) <= 2, message);
    }

    void TestSize(TestGpu& gpu, PostProcessing& post, UINT width, UINT height)
    {
        GBuffer buffers;
        buffers.Build(gpu.Device.Get(), width, height);
        const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
        auto depthDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R24G8_TYPELESS, width, height,
            1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        D3D12_CLEAR_VALUE depthClear = {};
        depthClear.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
        depthClear.DepthStencil.Depth = 0.6f;
        ComPtr<ID3D12Resource> depth;
        ThrowIfFailed(gpu.Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &depthDesc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE, &depthClear, IID_PPV_ARGS(&depth)));
        ComPtr<ID3D12DescriptorHeap> depthHeap, outputHeap;
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        heapDesc.NumDescriptors = 1;
        ThrowIfFailed(gpu.Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&depthHeap)));
        D3D12_DEPTH_STENCIL_VIEW_DESC dsvDesc = {};
        dsvDesc.Format = depthClear.Format;
        dsvDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        gpu.Device->CreateDepthStencilView(depth.Get(), &dsvDesc, depthHeap->GetCPUDescriptorHandleForHeapStart());
        gpu.Commands->ClearDepthStencilView(depthHeap->GetCPUDescriptorHandleForHeapStart(),
            D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.6f, 0, 0, nullptr);
        auto depthBarrier = CD3DX12_RESOURCE_BARRIER::Transition(depth.Get(),
            D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        gpu.Commands->ResourceBarrier(1, &depthBarrier);

        // Use the G-buffer's optimized clear values to keep GPU validation warning-free.
        // Its black albedo, +Z normal and depth 0.6 have known debug outputs.
        for (UINT i = 0; i < 2; ++i)
        {
            auto* resource = i == 0 ? buffers.AlbedoResource() : buffers.NormalResource();
            const auto rtv = i == 0 ? buffers.AlbedoRtv() : buffers.NormalRtv();
            auto toRtv = CD3DX12_RESOURCE_BARRIER::Transition(resource,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            gpu.Commands->ResourceBarrier(1, &toRtv);
            const float clear[4] = { i == 0 ? 0.0f : 0.5f, i == 0 ? 0.0f : 0.5f, 0, 1 };
            gpu.Commands->ClearRenderTargetView(rtv, clear, 0, nullptr);
            auto toSrv = CD3DX12_RESOURCE_BARRIER::Transition(resource,
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            gpu.Commands->ResourceBarrier(1, &toSrv);
        }

        const auto outputDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_TYPELESS,
            width, height, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        ComPtr<ID3D12Resource> output;
        ThrowIfFailed(gpu.Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &outputDesc,
            D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&output)));
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        ThrowIfFailed(gpu.Device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&outputHeap)));
        D3D12_RENDER_TARGET_VIEW_DESC rtvDesc = {};
        rtvDesc.Format = PostProcessing::OutputFormat;
        rtvDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        const auto outputRtv = outputHeap->GetCPUDescriptorHandleForHeapStart();
        gpu.Device->CreateRenderTargetView(output.Get(), &rtvDesc, outputRtv);
        post.Resize(gpu.Device.Get(), width, height, buffers.AlbedoResource(), buffers.NormalResource(), depth.Get());

        CD3DX12_ROOT_PARAMETER parameter;
        parameter.InitAsConstants(4, 2);
        CD3DX12_ROOT_SIGNATURE_DESC rootDesc(1, &parameter);
        ComPtr<ID3DBlob> rootBlob, errors;
        ThrowIfFailed(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, &errors));
        ComPtr<ID3D12RootSignature> fixtureRoot;
        ThrowIfFailed(gpu.Device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&fixtureRoot)));
        auto vs = d3dUtil::CompileShader(L"Shaders\\postprocess.hlsl", nullptr, "FullscreenVS", "vs_5_0");
        auto ps = d3dUtil::CompileShader(L"tests\\PostProcessFixture.hlsl", nullptr, "FixturePS", "ps_5_0");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline = {};
        pipeline.pRootSignature = fixtureRoot.Get();
        pipeline.VS = { vs->GetBufferPointer(), vs->GetBufferSize() };
        pipeline.PS = { ps->GetBufferPointer(), ps->GetBufferSize() };
        pipeline.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        pipeline.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        pipeline.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        pipeline.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        pipeline.DepthStencilState.DepthEnable = FALSE;
        pipeline.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        pipeline.SampleMask = UINT_MAX;
        pipeline.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        pipeline.NumRenderTargets = 1;
        pipeline.RTVFormats[0] = PostProcessing::HdrFormat;
        pipeline.SampleDesc.Count = 1;
        ComPtr<ID3D12PipelineState> fixturePso;
        ThrowIfFailed(gpu.Device->CreateGraphicsPipelineState(&pipeline, IID_PPV_ARGS(&fixturePso)));

        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
        UINT64 bytes;
        gpu.Device->GetCopyableFootprints(&outputDesc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
        auto readback = gpu.Readback(bytes);
        auto render = [&](UINT mode, const PostProcessSettings& settings, UINT debug = 0)
        {
            auto* commands = gpu.Commands.Get();
            post.BeginScene(commands);
            const D3D12_VIEWPORT viewport = { 0, 0, float(width), float(height), 0, 1 };
            const D3D12_RECT rect = { 0, 0, LONG(width), LONG(height) };
            commands->RSSetViewports(1, &viewport);
            commands->RSSetScissorRects(1, &rect);
            commands->SetGraphicsRootSignature(fixtureRoot.Get());
            commands->SetPipelineState(fixturePso.Get());
            const UINT constants[4] = { mode, width, height, 0 };
            commands->SetGraphicsRoot32BitConstants(0, 4, constants, 0);
            commands->IASetVertexBuffers(0, 0, nullptr);
            commands->IASetIndexBuffer(nullptr);
            commands->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            commands->DrawInstanced(4, 1, 0, 0);
            post.Apply(commands, outputRtv, settings, debug);
            auto toCopy = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),
                D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->ResourceBarrier(1, &toCopy);
            const CD3DX12_TEXTURE_COPY_LOCATION source(output.Get(), 0);
            const CD3DX12_TEXTURE_COPY_LOCATION target(readback.Get(), footprint);
            commands->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
            auto toRtv = CD3DX12_RESOURCE_BARRIER::Transition(output.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            commands->ResourceBarrier(1, &toRtv);
            gpu.Submit();
            const D3D12_RANGE range = { 0, SIZE_T(bytes) };
            void* mapped = nullptr;
            ThrowIfFailed(readback->Map(0, &range, &mapped));
            std::vector<unsigned char> pixels(width * height * 4);
            for (UINT y = 0; y < height; ++y)
                memcpy(pixels.data() + y * width * 4, static_cast<char*>(mapped) + footprint.Offset + y * footprint.Footprint.RowPitch, width * 4);
            const D3D12_RANGE noWrite = { 0, 0 };
            readback->Unmap(0, &noWrite);
            return pixels;
        };

        PostProcessSettings settings;
        settings.Bloom = settings.Vignette = false;
        settings.Exposure = 1.0f;
        const auto solid = render(0, settings);
        for (size_t i = 0; i < solid.size(); i += 4)
        {
            Near(solid[i], Encode(0.25f / 1.25f), "Quad gap/seam or incorrect tone mapping/sRGB (red)");
            Near(solid[i + 1], Encode(0.5f / 1.5f), "Incorrect tone mapping/sRGB (green)");
            Near(solid[i + 2], Encode(0.5f), "Incorrect tone mapping/sRGB (blue)");
            Require(solid[i + 3] == 255, "Full-screen quad failed to cover a pixel");
        }
        if (width > 1)
        {
            const auto gradient = render(2, settings);
            Require(gradient[0] < gradient[(width - 1) * 4], "Quad U coordinates are reversed");
            Require(gradient[1] < gradient[(height - 1) * width * 4 + 1], "Quad V coordinates are reversed");
            settings.Vignette = true;
            const auto vignette = render(0, settings);
            const size_t center = ((height / 2) * width + width / 2) * 4;
            Near(vignette[center], solid[center], "Vignette should preserve the center");
            Require(vignette[0] + 15 < solid[0], "Vignette did not darken the corners");
            settings.Vignette = false;
            const auto withoutBloom = render(1, settings);
            settings.Bloom = true;
            const auto bloom = render(1, settings);
            const size_t neighbor = ((height / 2) * width + width / 2 + 7) * 4;
            Require(bloom[neighbor] > withoutBloom[neighbor] + 5, "Bloom did not spread the HDR highlight");
            settings.Bloom = false;
            Require(render(1, settings) == withoutBloom, "Disabled bloom leaked previous frame data");
        }
        settings.Bloom = settings.Vignette = true;
        for (UINT debug = 1; debug <= 3; ++debug)
        {
            const auto result = render(1, settings, debug);
            Near(result[0], debug == 1 ? 0 : debug == 2 ? 128 : 153, "G-buffer debug input or effect bypass failed");
            Near(result[2], debug == 1 ? 0 : debug == 2 ? 255 : 153, "Normal/depth G-buffer input failed");
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        TestGpu gpu(argc > 1 && std::string(argv[1]) == "--hardware");
        PostProcessing post;
        post.Initialize(gpu.Device.Get());
        TestSize(gpu, post, 65, 49);
        TestSize(gpu, post, 1, 1);
        TestSize(gpu, post, 127, 73);
        std::cout << "PASS: full-screen quad coverage/UV, HDR tone mapping, single sRGB encoding, bloom, vignette, "
            << "toggles, all G-buffer inputs, odd-size/1x1 resize; D3D12 GPU validation clean.\n";
        return 0;
    }
    catch (const DxException& error) { std::wcerr << error.ToString() << L'\n'; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; }
    return 1;
}
