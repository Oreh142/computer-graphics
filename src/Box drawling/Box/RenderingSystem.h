#pragma once

#include "../../Common/MathHelper.h"

#include <vector>

using namespace DirectX;

struct PointLight
{
    XMFLOAT3 Position = XMFLOAT3(0.0f, 2.0f, 0.0f);
    float Radius = 8.0f;
    XMFLOAT3 Color = XMFLOAT3(1.0f, 1.0f, 1.0f);
    float Intensity = 2.5f;
};

struct DirectionalLight
{
    XMFLOAT3 Direction = XMFLOAT3(0.0f, -1.0f, 0.0f);
    float Intensity = 0.25f;
    XMFLOAT3 Color = XMFLOAT3(1.0f, 1.0f, 1.0f);
    float Padding = 0.0f;
};

struct SpotLight
{
    XMFLOAT3 Position = XMFLOAT3(0.0f, 4.0f, 0.0f);
    float Radius = 18.0f;
    XMFLOAT3 Direction = XMFLOAT3(0.0f, -1.0f, 0.0f);
    float OuterCos = 0.75f;
    XMFLOAT3 Color = XMFLOAT3(1.0f, 0.9f, 0.8f);
    float InnerCos = 0.90f;
    float Intensity = 6.0f;
    XMFLOAT3 Padding = XMFLOAT3(0.0f, 0.0f, 0.0f);
};


struct DeferredPassConstants
{
    XMFLOAT4X4 InvView = MathHelper::Identity4x4();
    XMFLOAT4X4 InvProj = MathHelper::Identity4x4();
    XMFLOAT3 CameraPosW = XMFLOAT3(0.0f, 0.0f, 0.0f);
    UINT PointLightCount = 0;
    UINT DirectionalLightCount = 0;
    UINT SpotLightCount = 0;
    UINT DebugView = 0;
    float Ambient = 0.001f;
    PointLight PointLights[16];
    DirectionalLight DirectionalLights[8];
    SpotLight SpotLights[8];
};

static_assert(sizeof(PointLight) == 32, "PointLight must match HLSL layout.");
static_assert(sizeof(DirectionalLight) == 32, "DirectionalLight must match HLSL layout.");
static_assert(sizeof(SpotLight) == 64, "SpotLight must match HLSL layout.");

class RenderingSystem
{
public:
    std::vector<PointLight> PointLights;
    std::vector<DirectionalLight> DirectionalLights;
    std::vector<SpotLight> SpotLights;
};
