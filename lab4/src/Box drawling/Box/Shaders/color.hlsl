//***************************************************************************************
// color.hlsl
//
// Textured geometry with simple directional lighting.
// Vertex input: position + normal + UV.
// gTexTransform enables UV tiling / scroll animation.
// gLightDir is the unit vector pointing FROM surface TOWARDS the light.
//***************************************************************************************

cbuffer cbPerObject : register(b0)
{
    float4x4 gWorldViewProj;  // MVP (row-major, transposed before upload)
    float4x4 gWorld;          // World matrix for normal transform
    float4x4 gTexTransform;   // UV tiling / scroll
    float4   gDiffuseAlbedo;  // material colour tint
    float3   gLightDir;       // unit vector towards the light (world space)
    float    gPad0;
    float3   gLightColor;     // light RGB colour
    float    gPad1;
};

Texture2D    gDiffuseMap     : register(t0);
SamplerState gSamAnisotropic : register(s0);

struct VertexIn
{
    float3 PosL    : POSITION;
    float3 NormalL : NORMAL;
    float2 TexC    : TEXCOORD;
};

struct VertexOut
{
    float4 PosH    : SV_POSITION;
    float3 NormalW : NORMAL;
    float2 TexC    : TEXCOORD;
};

VertexOut VS(VertexIn vin)
{
    VertexOut vout;

    vout.PosH    = mul(float4(vin.PosL, 1.0f), gWorldViewProj);
    vout.NormalW = mul(vin.NormalL, (float3x3)gWorld);

    float4 texC = mul(float4(vin.TexC, 0.0f, 1.0f), gTexTransform);
    vout.TexC   = texC.xy;

    return vout;
}

float4 PS(VertexOut pin) : SV_Target
{
    float3 N = normalize(pin.NormalW);

    // Lambert diffuse
    float  diff       = saturate(dot(N, gLightDir));
    float3 ambient    = float3(0.15f, 0.15f, 0.15f);
    float3 lightFinal = gLightColor * diff + ambient;

    float4 texColor = gDiffuseMap.Sample(gSamAnisotropic, pin.TexC);

    return float4(texColor.rgb * lightFinal * gDiffuseAlbedo.rgb,
                  texColor.a  * gDiffuseAlbedo.a);
}
