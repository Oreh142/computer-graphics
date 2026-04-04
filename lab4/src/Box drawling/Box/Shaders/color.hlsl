cbuffer cbPerObject : register(b0)
{
    float4x4 gWorld;
    float4x4 gWorldViewProj;
    float2 gUvScale;
    float2 gUvOffset;
    float4 gTint;
};

Texture2D gDiffuseMap : register(t0);
SamplerState gsamLinearWrap : register(s0);

struct VertexIn
{
    float3 PosL : POSITION;
    float3 NormalL : NORMAL;
    float2 TexC : TEXCOORD;
};

struct GBufferOut
{
    float4 Albedo : SV_Target0;
    float4 NormalDepth : SV_Target1;
};

struct VertexOut
{
    float4 PosH : SV_POSITION;
    float3 NormalW : NORMAL;
    float2 TexC : TEXCOORD;
};

VertexOut GBufferVS(VertexIn vin)
{
    VertexOut vout;
    float4 posW = mul(float4(vin.PosL, 1.0f), gWorld);
    vout.PosH = mul(float4(vin.PosL, 1.0f), gWorldViewProj);
    vout.NormalW = normalize(mul(vin.NormalL, (float3x3)gWorld));
    vout.TexC = vin.TexC * gUvScale + gUvOffset;
    return vout;
}

GBufferOut GBufferPS(VertexOut pin)
{
    GBufferOut o;
    float4 albedo = gDiffuseMap.Sample(gsamLinearWrap, pin.TexC) * gTint;
    o.Albedo = albedo;
    o.NormalDepth = float4(normalize(pin.NormalW) * 0.5f + 0.5f, pin.PosH.z);
    return o;
}

struct FSOut
{
    float4 PosH : SV_POSITION;
    float2 Uv : TEXCOORD;
};

FSOut LightingVS(uint vid : SV_VertexID)
{
    FSOut o;
    float2 pos = float2((vid == 2) ? 3.0f : -1.0f, (vid == 1) ? 3.0f : -1.0f);
    o.PosH = float4(pos, 0.0f, 1.0f);
    o.Uv = float2(0.5f * (pos.x + 1.0f), 1.0f - 0.5f * (pos.y + 1.0f));
    return o;
}

Texture2D gAlbedoMap : register(t0);
Texture2D gNormalDepthMap : register(t1);

#define MAX_POINT_LIGHTS 16
#define MAX_DIRECTIONAL_LIGHTS 8
#define MAX_SPOT_LIGHTS 8

struct PointLight
{
    float3 Position;
    float Radius;
    float3 Color;
    float Intensity;
};

struct DirectionalLight
{
    float3 Direction;
    float Intensity;
    float3 Color;
    float pad0;
};

struct SpotLight
{
    float3 Position;
    float Radius;
    float3 Direction;
    float OuterCos;
    float3 Color;
    float InnerCos;
    float Intensity;
    float3 pad1;
};

cbuffer cbLighting : register(b1)
{
    float4x4 gInvView;
    float4x4 gInvProj;
    float3 gCameraPosW;
    uint gPointLightCount;
    uint gDirectionalLightCount;
    uint gSpotLightCount;
    float gAmbient;
    float gPadding0;

    PointLight gPointLights[MAX_POINT_LIGHTS];
    DirectionalLight gDirectionalLights[MAX_DIRECTIONAL_LIGHTS];
    SpotLight gSpotLights[MAX_SPOT_LIGHTS];
};

float3 ReconstructWorldPos(float2 uv, float depthNdc)
{
    float ndcX = uv.x * 2.0f - 1.0f;
    float ndcY = 1.0f - uv.y * 2.0f;
    float4 posV = mul(float4(ndcX, ndcY, depthNdc, 1.0f), gInvProj);
    posV /= max(posV.w, 1e-6f);
    float4 posW = mul(posV, gInvView);
    return posW.xyz;
}

float3 EvaluateDirectionalLight(float3 albedo, float3 normalW, DirectionalLight light)
{
    const float3 L = normalize(-light.Direction);
    const float ndotl = saturate(dot(normalW, L));
    return albedo * light.Color * (light.Intensity * ndotl);
}

float3 EvaluatePointLight(float3 albedo, float3 normalW, float3 posW, PointLight light)
{
    const float3 toLight = light.Position - posW;
    const float distSq = dot(toLight, toLight);
    const float dist = sqrt(max(distSq, 1e-6f));
    const float3 L = toLight / dist;

    const float ndotl = saturate(dot(normalW, L));
    const float rangeT = saturate(1.0f - dist / max(light.Radius, 1e-3f));
    const float smoothRangeAtt = rangeT * rangeT;
    const float invDistAtt = 1.0f / (1.0f + distSq);
    const float attenuation = smoothRangeAtt * invDistAtt;

    return albedo * light.Color * (light.Intensity * ndotl * attenuation);
}

float3 EvaluateSpotLight(float3 albedo, float3 normalW, float3 posW, SpotLight light)
{
    const float3 toLight = light.Position - posW;
    const float distSq = dot(toLight, toLight);
    const float dist = sqrt(max(distSq, 1e-6f));
    const float3 L = toLight / dist;

    const float3 spotDir = normalize(-light.Direction);
    const float coneDot = dot(spotDir, L);
    const float coneAtt = saturate((coneDot - light.OuterCos) / max(light.InnerCos - light.OuterCos, 1e-3f));
    const float rangeT = saturate(1.0f - dist / max(light.Radius, 1e-3f));
    const float smoothRangeAtt = rangeT * rangeT;
    const float invDistAtt = 1.0f / (1.0f + distSq);
    const float attenuation = coneAtt * smoothRangeAtt * invDistAtt;

    const float ndotl = saturate(dot(normalW, L));
    return albedo * light.Color * (light.Intensity * ndotl * attenuation);
}

float4 LightingPS(FSOut pin) : SV_Target
{
    float4 albedo = gAlbedoMap.Sample(gsamLinearWrap, pin.Uv);
    float4 nd = gNormalDepthMap.Sample(gsamLinearWrap, pin.Uv);

    float3 normalW = normalize(nd.xyz * 2.0f - 1.0f);
    float depthNdc = nd.w;
    float3 posW = ReconstructWorldPos(pin.Uv, depthNdc);

    float3 result = albedo.rgb * gAmbient;

    [loop]
    for (uint i = 0; i < gDirectionalLightCount; ++i)
    {
        result += EvaluateDirectionalLight(albedo.rgb, normalW, gDirectionalLights[i]);
    }

    [loop]
    for (uint i = 0; i < gPointLightCount; ++i)
    {
        result += EvaluatePointLight(albedo.rgb, normalW, posW, gPointLights[i]);
    }

    [loop]
    for (uint i = 0; i < gSpotLightCount; ++i)
    {
        result += EvaluateSpotLight(albedo.rgb, normalW, posW, gSpotLights[i]);
    }

    return float4(result, albedo.a);
}
