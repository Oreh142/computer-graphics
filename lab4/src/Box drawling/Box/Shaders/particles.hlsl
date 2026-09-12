struct Particle
{
    float3 Position;
    float Age;
    float3 Velocity;
    float Lifetime;
    float3 Color;
    float Radius;
};

cbuffer Simulation : register(b0)
{
    float gDeltaTime;
    uint gEmitCount;
    uint gSeed;
    uint gCapacity;
    float3 gEmitter;
    float gFloorY;
    float3 gAcceleration;
    float gSimulationPadding;
};

ConsumeStructuredBuffer<Particle> gInput : register(u0);
AppendStructuredBuffer<Particle> gOutput : register(u1);
// A stable copy is essential: Consume changes the input UAV counter in parallel.
ByteAddressBuffer gCountSnapshot : register(t1);

[numthreads(256, 1, 1)]
void SimulateCS(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= min(gCountSnapshot.Load(0), gCapacity))
        return;

    Particle p = gInput.Consume();
    p.Age += gDeltaTime;
    // Constant acceleration, integrated entirely on the GPU.
    p.Position += p.Velocity * gDeltaTime + 0.5f * gAcceleration * gDeltaTime * gDeltaTime;
    p.Velocity += gAcceleration * gDeltaTime;
    if (p.Age < p.Lifetime && p.Position.y - p.Radius > gFloorY)
        gOutput.Append(p);
}

uint Hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float Random01(inout uint state)
{
    state = Hash(state);
    return (state & 0x00ffffffu) / 16777216.0f;
}

[numthreads(256, 1, 1)]
void EmitCS(uint3 threadId : SV_DispatchThreadID)
{
    // The snapshot now contains the survivor count after SimulateCS.
    uint freeSlots = gCapacity - min(gCountSnapshot.Load(0), gCapacity);
    if (threadId.x >= min(gEmitCount, freeSlots))
        return;

    uint rng = Hash(threadId.x + gSeed * 0x9e3779b9u);
    float angle = Random01(rng) * 6.2831853f;
    float radialSpeed = 0.6f + 1.8f * Random01(rng);
    Particle p;
    p.Position = gEmitter + float3(cos(angle), 0.0f, sin(angle)) * (0.12f * Random01(rng));
    p.Age = 0.0f;
    p.Velocity = float3(cos(angle) * radialSpeed, 3.5f + 1.5f * Random01(rng), sin(angle) * radialSpeed);
    p.Lifetime = 2.0f + 2.0f * Random01(rng);
    p.Color = lerp(float3(0.08f, 0.45f, 1.0f), float3(1.0f, 0.55f, 0.08f), Random01(rng));
    p.Radius = 0.025f + 0.035f * Random01(rng);
    gOutput.Append(p);
}

cbuffer Camera : register(b1)
{
    float4x4 gViewProjection;
    float3 gCameraRight;
    float gCameraPadding0;
    float3 gCameraUp;
    float gCameraPadding1;
};

StructuredBuffer<Particle> gParticles : register(t0);

struct PointOut
{
    float3 Position : POSITION;
    float Radius : PSIZE;
    float3 Color : COLOR;
};

PointOut ParticleVS(uint vertexId : SV_VertexID)
{
    Particle p = gParticles[vertexId];
    PointOut result;
    result.Position = p.Position;
    result.Radius = p.Radius;
    result.Color = p.Color;
    return result;
}

struct BillboardOut
{
    float4 Position : SV_POSITION;
    float2 Corner : TEXCOORD;
    float3 Color : COLOR;
};

[maxvertexcount(4)]
void ParticleGS(point PointOut input[1], inout TriangleStream<BillboardOut> stream)
{
    const float2 corners[4] =
    {
        float2(-1.0f, -1.0f), float2(-1.0f, 1.0f),
        float2(1.0f, -1.0f), float2(1.0f, 1.0f)
    };
    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        BillboardOut output;
        float3 world = input[0].Position + input[0].Radius *
            (corners[i].x * gCameraRight + corners[i].y * gCameraUp);
        output.Position = mul(float4(world, 1.0f), gViewProjection);
        output.Corner = corners[i];
        output.Color = input[0].Color;
        stream.Append(output);
    }
    stream.RestartStrip();
}

struct ParticleGBuffer
{
    float4 Albedo : SV_Target0;
    float2 Normal : SV_Target1;
};

ParticleGBuffer ParticlePS(BillboardOut input)
{
    float radiusSquared = dot(input.Corner, input.Corner);
    clip(1.0f - radiusSquared);
    // A curved normal makes an opaque disc look like a small bead.
    float3 normal = normalize(gCameraRight * input.Corner.x + gCameraUp * input.Corner.y -
        cross(gCameraRight, gCameraUp) * sqrt(saturate(1.0f - radiusSquared)));
    normal /= abs(normal.x) + abs(normal.y) + abs(normal.z);
    float2 encoded = normal.xy;
    if (normal.z < 0.0f)
    {
        float2 signs = float2(encoded.x >= 0.0f ? 1.0f : -1.0f, encoded.y >= 0.0f ? 1.0f : -1.0f);
        encoded = (1.0f - abs(encoded.yx)) * signs;
    }
    ParticleGBuffer result;
    result.Albedo = float4(input.Color, 1.0f);
    result.Normal = encoded * 0.5f + 0.5f;
    return result;
}
