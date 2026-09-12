#include "../Shaders/postprocess.hlsl"

cbuffer Fixture : register(b2)
{
    uint gFixtureMode;
    uint gFixtureWidth;
    uint gFixtureHeight;
    uint gFixturePadding;
};

float4 FixturePS(FullscreenOut input) : SV_Target
{
    if (gFixtureMode == 0) return float4(0.25f, 0.5f, 1.0f, 1.0f);
    if (gFixtureMode == 2) return float4(input.Uv, 0.25f, 1.0f);
    float2 distanceToCenter = abs(input.Position.xy - 0.5f * float2(gFixtureWidth, gFixtureHeight));
    return float4((distanceToCenter.x < 3.0f && distanceToCenter.y < 3.0f) ? 8.0f.xxx : 0.03f.xxx, 1.0f);
}
