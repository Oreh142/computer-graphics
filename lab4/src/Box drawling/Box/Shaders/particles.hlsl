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
    float4x4 gCollisionViewProjection;
    float4x4 gCollisionInverseViewProjection;
    float3 gCollisionCameraPosition;
    float gCollisionThickness;
    uint2 gDepthDimensions;
    uint gCollisionEnabled;
    float gCollisionRestitution;
};

ConsumeStructuredBuffer<Particle> gInput : register(u0);
AppendStructuredBuffer<Particle> gOutput : register(u1);
// A stable copy is essential: Consume changes the input UAV counter in parallel.
ByteAddressBuffer gCountSnapshot : register(t1);
Texture2D<float> gSceneDepth : register(t2);

bool ProjectToDepth(float3 worldPosition, out float2 uv, out float deviceDepth)
{
    float4 clip = mul(float4(worldPosition, 1.0f), gCollisionViewProjection);
    if (clip.w <= 0.00001f)
        return false;

    float3 ndc = clip.xyz / clip.w;
    uv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
    deviceDepth = ndc.z;
    return all(uv >= 0.0f) && all(uv <= 1.0f) && deviceDepth >= 0.0f && deviceDepth <= 1.0f;
}

float3 ReconstructWorld(int2 pixel, float deviceDepth)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(gDepthDimensions);
    float4 world = mul(float4(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f,
        deviceDepth, 1.0f), gCollisionInverseViewProjection);
    return world.xyz / world.w;
}

void ResolveDepthCollision(float3 oldPosition, inout Particle p)
{
    if (gCollisionEnabled == 0)
        return;

    float2 uv;
    float particleDepth;
    if (!ProjectToDepth(p.Position, uv, particleDepth))
        return;

    int2 maxPixel = int2(gDepthDimensions) - 1;
    int2 pixel = clamp(int2(uv * float2(gDepthDimensions)), int2(0, 0), maxPixel);
    float centerDepth = gSceneDepth.Load(int3(pixel, 0));
    // A cleared depth value means that no visible surface occupies this pixel.
    if (centerDepth >= 0.999999f)
        return;

    int2 leftPixel = max(pixel - int2(1, 0), int2(0, 0));
    int2 rightPixel = min(pixel + int2(1, 0), maxPixel);
    int2 upPixel = max(pixel - int2(0, 1), int2(0, 0));
    int2 downPixel = min(pixel + int2(0, 1), maxPixel);
    float leftDepth = gSceneDepth.Load(int3(leftPixel, 0));
    float rightDepth = gSceneDepth.Load(int3(rightPixel, 0));
    float upDepth = gSceneDepth.Load(int3(upPixel, 0));
    float downDepth = gSceneDepth.Load(int3(downPixel, 0));

    float3 center = ReconstructWorld(pixel, centerDepth);
    float3 left = ReconstructWorld(leftPixel, leftDepth);
    float3 right = ReconstructWorld(rightPixel, rightDepth);
    float3 up = ReconstructWorld(upPixel, upDepth);
    float3 down = ReconstructWorld(downPixel, downDepth);

    // At silhouettes, choose the neighbour whose depth is closest to the center.
    // Both tangents keep the +screen-x/+screen-y direction, so cross(x, y)
    // initially points toward the camera for a front-facing surface.
    float3 tangentX = abs(rightDepth - centerDepth) < abs(centerDepth - leftDepth)
        ? right - center : center - left;
    float3 tangentY = abs(downDepth - centerDepth) < abs(centerDepth - upDepth)
        ? down - center : center - up;
    float3 normal = cross(tangentX, tangentY);
    float normalLengthSquared = dot(normal, normal);
    if (normalLengthSquared < 0.0000000001f)
        return;
    normal *= rsqrt(normalLengthSquared);
    if (dot(normal, gCollisionCameraPosition - center) < 0.0f)
        normal = -normal;

    float oldDistance = dot(oldPosition - center, normal);
    float newDistance = dot(p.Position - center, normal);
    float normalVelocity = dot(p.Velocity, normal);
    float contactDistance = p.Radius + gCollisionThickness;
    if (normalVelocity >= 0.0f || newDistance > contactDistance ||
        oldDistance < -gCollisionThickness)
        return;

    // Find the center position at first contact and push it just outside the surface.
    float distanceChange = oldDistance - newDistance;
    float hitTime = distanceChange > 0.00001f
        ? saturate((oldDistance - contactDistance) / distanceChange) : 0.0f;
    p.Position = lerp(oldPosition, p.Position, hitTime);
    p.Position += normal * max(0.0f, contactDistance - dot(p.Position - center, normal));

    // Reflect the normal component and damp the tangent to imitate friction.
    float3 normalPart = normalVelocity * normal;
    float3 tangentPart = p.Velocity - normalPart;
    p.Velocity = tangentPart * 0.88f - normalPart * gCollisionRestitution;
}

[numthreads(256, 1, 1)]
void SimulateCS(uint3 threadId : SV_DispatchThreadID)
{
    if (threadId.x >= min(gCountSnapshot.Load(0), gCapacity))
        return;

    Particle p = gInput.Consume();
    p.Age += gDeltaTime;
    // Constant acceleration, integrated entirely on the GPU.
    float3 oldPosition = p.Position;
    p.Position += p.Velocity * gDeltaTime + 0.5f * gAcceleration * gDeltaTime * gDeltaTime;
    p.Velocity += gAcceleration * gDeltaTime;
    ResolveDepthCollision(oldPosition, p);
    // With depth collision enabled, keep particles alive around the old floor cutoff
    // so they can visibly bounce from the rendered floor instead of being deleted there.
    float killY = gCollisionEnabled != 0 ? gFloorY - 2.0f : gFloorY;
    if (p.Age < p.Lifetime && p.Position.y - p.Radius > killY)
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
