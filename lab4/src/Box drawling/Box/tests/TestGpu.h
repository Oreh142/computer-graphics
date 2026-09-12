#pragma once

#include "../../../Common/d3dUtil.h"
#include <d3d12sdklayers.h>
#include <iostream>
#include <stdexcept>

namespace GpuTests
{
    using Microsoft::WRL::ComPtr;
    void Require(bool condition, const char* message)
    {
        if (!condition)
            throw std::runtime_error(message);
    }

    struct TestGpu
    {
        ComPtr<ID3D12Device> Device;
        ComPtr<ID3D12CommandQueue> Queue;
        ComPtr<ID3D12CommandAllocator> Allocator;
        ComPtr<ID3D12GraphicsCommandList> Commands;
        ComPtr<ID3D12Fence> Fence;
        ComPtr<ID3D12InfoQueue> Info;
        UINT64 FenceValue = 0;

        explicit TestGpu(bool hardware)
        {
            ComPtr<ID3D12Debug> debug;
            ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)));
            debug->EnableDebugLayer();
            ComPtr<ID3D12Debug1> validation;
            ThrowIfFailed(debug.As(&validation));
            validation->SetEnableGPUBasedValidation(TRUE);
            ComPtr<IDXGIFactory4> factory;
            ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));
            ComPtr<IDXGIAdapter> adapter;
            if (!hardware)
                ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
            ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&Device)));
            ThrowIfFailed(Device.As(&Info));
            D3D12_COMMAND_QUEUE_DESC queue = {};
            ThrowIfFailed(Device->CreateCommandQueue(&queue, IID_PPV_ARGS(&Queue)));
            ThrowIfFailed(Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&Allocator)));
            ThrowIfFailed(Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                Allocator.Get(), nullptr, IID_PPV_ARGS(&Commands)));
            ThrowIfFailed(Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence)));
        }

        void Submit()
        {
            ThrowIfFailed(Commands->Close());
            ID3D12CommandList* lists[] = { Commands.Get() };
            Queue->ExecuteCommandLists(1, lists);
            ThrowIfFailed(Queue->Signal(Fence.Get(), ++FenceValue));
            HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            Require(event != nullptr, "CreateEvent failed");
            const HRESULT eventResult = Fence->SetEventOnCompletion(FenceValue, event);
            if (FAILED(eventResult))
            {
                CloseHandle(event);
                ThrowIfFailed(eventResult);
            }
            const DWORD wait = WaitForSingleObject(event, 30000);
            CloseHandle(event);
            Require(wait == WAIT_OBJECT_0, "GPU wait failed or timed out");
            ThrowIfFailed(Device->GetDeviceRemovedReason());
            bool validationError = false;
            for (UINT64 i = 0; i < Info->GetNumStoredMessages(); ++i)
            {
                SIZE_T size = 0;
                Info->GetMessage(i, nullptr, &size);
                std::vector<char> storage(size);
                auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                ThrowIfFailed(Info->GetMessage(i, message, &size));
                if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                {
                    std::cerr << message->pDescription << '\n';
                    validationError = true;
                }
            }
            Info->ClearStoredMessages();
            Require(!validationError, "D3D12 validation reported warnings/errors");
            ThrowIfFailed(Allocator->Reset());
            ThrowIfFailed(Commands->Reset(Allocator.Get(), nullptr));
        }

        ComPtr<ID3D12Resource> Readback(UINT64 size)
        {
            const auto heap = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
            const auto desc = CD3DX12_RESOURCE_DESC::Buffer(size);
            ComPtr<ID3D12Resource> result;
            ThrowIfFailed(Device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
                &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&result)));
            return result;
        }
    };

}
