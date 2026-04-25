cbuffer cbPerObject : register(b0)
{
    float4x4 gWorld;
    float4x4 gViewProj;
    float2 gUvScale;
    float2 gUvOffset;
    float4 gTint;
    float3 gObjectCameraPosW;
    float gDisplacementScale;
    float gTessMinDistance;
    float gTessMaxDistance;
    float gTessMinFactor;
    float gTessMaxFactor;
};

Texture2D gDiffuseMap : register(t0);
Texture2D gNormalMap : register(t1);
Texture2D gDisplacementMap : register(t2);
SamplerState gsamLinearWrap : register(s0);

struct VertexIn
{
    float3 PosL : POSITION;
    float3 NormalL : NORMAL;
    float3 TangentL : TANGENT;
    float2 TexC : TEXCOORD;
};

struct SurfaceOut
{
    float4 PosH : SV_POSITION;
    float3 NormalW : NORMAL;
    float3 TangentW : TANGENT;
    float2 TexC : TEXCOORD;
};

struct GBufferOut
{
    float4 Albedo : SV_Target0;
    float2 NormalRG : SV_Target1;
};

float2 EncodeOctahedron(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 enc = n.xy;
    if (n.z < 0.0f)
    {
        float2 signVal = float2(enc.x >= 0.0f ? 1.0f : -1.0f, enc.y >= 0.0f ? 1.0f : -1.0f);
        enc = (1.0f - abs(enc.yx)) * signVal;
    }
    return enc * 0.5f + 0.5f;
}

float3 DecodeOctahedron(float2 enc)
{
    float2 f = enc * 2.0f - 1.0f;
    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
    float t = saturate(-n.z);
    n.x += (n.x >= 0.0f) ? -t : t;
    n.y += (n.y >= 0.0f) ? -t : t;
    return normalize(n);
}

float3 SampleNormalTS(float2 uv)
{
    float3 normalTS = gNormalMap.Sample(gsamLinearWrap, uv).xyz * 2.0f - 1.0f;
    return normalize(normalTS);
}

float SampleDisplacement(float2 uv)
{
    float3 disp = gDisplacementMap.SampleLevel(gsamLinearWrap, uv, 0).rgb;
    return dot(disp, float3(0.299f, 0.587f, 0.114f));
}

float3 TangentToWorld(float3 normalTS, float3 normalW, float3 tangentW)
{
    tangentW = normalize(tangentW - normalW * dot(tangentW, normalW));
    float3 bitangentW = normalize(cross(normalW, tangentW));
    return normalize(normalTS.x * tangentW + normalTS.y * bitangentW + normalTS.z * normalW);
}

SurfaceOut BuildSurface(float3 posL, float3 normalL, float3 tangentL, float2 texC)
{
    SurfaceOut outv;
    float4 posW = mul(float4(posL, 1.0f), gWorld);
    outv.PosH = mul(posW, gViewProj);
    outv.NormalW = normalize(mul(normalL, (float3x3)gWorld));
    outv.TangentW = normalize(mul(tangentL, (float3x3)gWorld));
    outv.TexC = texC * gUvScale + gUvOffset;
    return outv;
}

SurfaceOut GBufferVS(VertexIn vin)
{
    return BuildSurface(vin.PosL, vin.NormalL, vin.TangentL, vin.TexC);
}

GBufferOut GBufferPS(SurfaceOut pin)
{
    GBufferOut o;
    float4 albedo = gDiffuseMap.Sample(gsamLinearWrap, pin.TexC) * gTint;
    float3 normalTS = SampleNormalTS(pin.TexC);
    float3 normalW = TangentToWorld(normalTS, normalize(pin.NormalW), normalize(pin.TangentW));
    o.Albedo = albedo;
    o.NormalRG = EncodeOctahedron(normalW);
    return o;
}

struct HullIn
{
    float3 PosL : POSITION;
    float3 NormalL : NORMAL;
    float3 TangentL : TANGENT;
    float2 TexC : TEXCOORD;
};

struct HullOut
{
    float3 PosL : POSITION;
    float3 NormalL : NORMAL;
    float3 TangentL : TANGENT;
    float2 TexC : TEXCOORD;
};

struct PatchTess
{
    float Edges[3] : SV_TessFactor;
    float Inside : SV_InsideTessFactor;
};

HullOut GBufferTessVS(VertexIn vin)
{
    HullOut hout;
    hout.PosL = vin.PosL;
    hout.NormalL = vin.NormalL;
    hout.TangentL = vin.TangentL;
    hout.TexC = vin.TexC;
    return hout;
}

float ComputeTessFactor(float3 posL)
{
    float3 posW = mul(float4(posL, 1.0f), gWorld).xyz;
    float dist = distance(posW, gObjectCameraPosW);
    float t = saturate((dist - gTessMinDistance) / max(gTessMaxDistance - gTessMinDistance, 1e-4f));
    return lerp(gTessMaxFactor, gTessMinFactor, t);
}

PatchTess CalcPatchConstants(InputPatch<HullOut, 3> patch, uint patchId : SV_PrimitiveID)
{
    PatchTess tess;

    tess.Edges[0] = ComputeTessFactor(0.5f * (patch[1].PosL + patch[2].PosL));
    tess.Edges[1] = ComputeTessFactor(0.5f * (patch[2].PosL + patch[0].PosL));
    tess.Edges[2] = ComputeTessFactor(0.5f * (patch[0].PosL + patch[1].PosL));
    tess.Inside = (tess.Edges[0] + tess.Edges[1] + tess.Edges[2]) / 3.0f;

    return tess;
}

[domain("tri")]
[partitioning("fractional_odd")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("CalcPatchConstants")]
[maxtessfactor(16.0f)]
HullOut GBufferHS(InputPatch<HullOut, 3> patch, uint cpId : SV_OutputControlPointID, uint patchId : SV_PrimitiveID)
{
    return patch[cpId];
}

[domain("tri")]
SurfaceOut GBufferDS(PatchTess patchTess, float3 bary : SV_DomainLocation, const OutputPatch<HullOut, 3> patch)
{
    float3 posL = patch[0].PosL * bary.x + patch[1].PosL * bary.y + patch[2].PosL * bary.z;
    float3 normalL = normalize(patch[0].NormalL * bary.x + patch[1].NormalL * bary.y + patch[2].NormalL * bary.z);
    float3 tangentL = normalize(patch[0].TangentL * bary.x + patch[1].TangentL * bary.y + patch[2].TangentL * bary.z);
    float2 texC = patch[0].TexC * bary.x + patch[1].TexC * bary.y + patch[2].TexC * bary.z;

    float2 uv = texC * gUvScale + gUvOffset;
    float height = SampleDisplacement(uv);
    posL += normalL * ((height - 0.5f) * gDisplacementScale);

    SurfaceOut outv;
    float4 posW = mul(float4(posL, 1.0f), gWorld);
    outv.PosH = mul(posW, gViewProj);
    outv.NormalW = normalize(mul(normalL, (float3x3)gWorld));
    outv.TangentW = normalize(mul(tangentL, (float3x3)gWorld));
    outv.TexC = uv;
    return outv;
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
Texture2D gNormalBufferMap : register(t1);
Texture2D gDepthMap : register(t2);

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
    uint gDebugView;
    float gAmbient;

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
    float3 L = normalize(-light.Direction);
    float ndotl = saturate(dot(normalW, L));
    return albedo * light.Color * (light.Intensity * ndotl);
}

float3 EvaluatePointLight(float3 albedo, float3 normalW, float3 posW, PointLight light)
{
    float3 toLight = light.Position - posW;
    float distSq = dot(toLight, toLight);
    float dist = sqrt(max(distSq, 1e-6f));
    float3 L = toLight / dist;

    float ndotl = saturate(dot(normalW, L));
    float rangeT = saturate(1.0f - dist / max(light.Radius, 1e-3f));
    float smoothRangeAtt = rangeT * rangeT;
    float invDistAtt = 1.0f / (1.0f + distSq);
    float attenuation = smoothRangeAtt * invDistAtt;

    return albedo * light.Color * (light.Intensity * ndotl * attenuation);
}

float3 EvaluateSpotLight(float3 albedo, float3 normalW, float3 posW, SpotLight light)
{
    float3 toLight = light.Position - posW;
    float distSq = dot(toLight, toLight);
    float dist = sqrt(max(distSq, 1e-6f));
    float3 L = toLight / dist;

    float3 spotDir = normalize(-light.Direction);
    float coneDot = dot(spotDir, L);
    float coneAtt = saturate((coneDot - light.OuterCos) / max(light.InnerCos - light.OuterCos, 1e-3f));
    float rangeT = saturate(1.0f - dist / max(light.Radius, 1e-3f));
    float smoothRangeAtt = rangeT * rangeT;
    float invDistAtt = 1.0f / (1.0f + distSq);
    float attenuation = coneAtt * smoothRangeAtt * invDistAtt;

    float ndotl = saturate(dot(normalW, L));
    return albedo * light.Color * (light.Intensity * ndotl * attenuation);
}

float4 LightingPS(FSOut pin) : SV_Target
{
    float4 albedo = gAlbedoMap.Sample(gsamLinearWrap, pin.Uv);
    float2 normalTex = gNormalBufferMap.Sample(gsamLinearWrap, pin.Uv).rg;
    float depthNdc = gDepthMap.Sample(gsamLinearWrap, pin.Uv).r;

    float3 normalW = DecodeOctahedron(normalTex);
    float3 posW = ReconstructWorldPos(pin.Uv, depthNdc);

    if (gDebugView == 1)
        return float4(albedo.rgb, 1.0f);
    if (gDebugView == 2)
        return float4(normalW * 0.5f + 0.5f, 1.0f);
    if (gDebugView == 3)
        return float4(depthNdc.xxx, 1.0f);

    float3 result = albedo.rgb * gAmbient;

    [loop]
    for (uint dirIndex = 0; dirIndex < gDirectionalLightCount; ++dirIndex)
    {
        result += EvaluateDirectionalLight(albedo.rgb, normalW, gDirectionalLights[dirIndex]);
    }

    [loop]
    for (uint pointIndex = 0; pointIndex < gPointLightCount; ++pointIndex)
    {
        result += EvaluatePointLight(albedo.rgb, normalW, posW, gPointLights[pointIndex]);
    }

    [loop]
    for (uint spotIndex = 0; spotIndex < gSpotLightCount; ++spotIndex)
    {
        result += EvaluateSpotLight(albedo.rgb, normalW, posW, gSpotLights[spotIndex]);
    }

    return float4(result, albedo.a);
}
