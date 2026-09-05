//=================================================================================================
//
//  Wavefront-style RayQuery path tracer.
//
//  This is intentionally a small first step away from the megakernel path: rays, path state, and
//  sun shadow work are stored in explicit queues, while each bounce is driven by separate dispatches.
//
//=================================================================================================

#include <DescriptorTables.hlsl>
#include <Constants.hlsl>
#include <Quaternion.hlsl>
#include <BRDF.hlsl>
#include <RayTracing.hlsl>
#include <Sampling.hlsl>

#include "SharedTypes.h"
#include "AppSettings.hlsl"

struct RayTraceConstants
{
    row_major float4x4 InvViewProjection;

    float3 SunDirectionWS;
    float CosSunAngularRadius;
    float3 SunIrradiance;
    float SinSunAngularRadius;
    float3 SunRenderColor;
    uint Padding;
    float3 CameraPosWS;
    uint CurrSampleIdx;
    uint TotalNumPixels;

    uint VtxBufferIdx;
    uint VtxFloatBufferIdx;
    uint IdxBufferIdx;
    uint GeometryInfoBufferIdx;
    uint MaterialBufferIdx;
    uint SkyTextureIdx;
    uint NumLights;

    uint myFlags;
    uint WavefrontReadQueue;
    uint WavefrontWriteQueue;
    uint WavefrontBounce;
    uint WavefrontPadding;
};

struct LightConstants
{
    SpotLight Lights[MaxSpotLights];
    float4x4 ShadowMatrices[MaxSpotLights];
};

struct PathState
{
    float3 Radiance;
    float3 Throughput;
    uint PixelIdx;
    uint SampleSetIdx;
    uint PathLength;
    uint Flags;
    float Roughness;
    float Padding;
};

struct RayWorkItem
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
    uint PathStateIdx;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

struct ShadowWorkItem
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
    float3 Contribution;
    uint PathStateIdx;
};

struct HitWorkItem
{
    uint PathStateIdx;
    uint GeometryIdx;
    uint PrimitiveIdx;
    float2 Bary;
    float3 RayOrigin;
    float RayT;
    float3 RayDirection;
    uint Padding;
};

struct HitInfoRQ
{
    bool Hit;
    float T;
    uint GeometryIdx;
    uint PrimitiveIdx;
    float2 Bary;
};

struct NextBounce
{
    bool Valid;
    float3 DirWS;
    float3 Throughput;
    float Roughness;
    bool IsDiffuse;
};

RaytracingAccelerationStructure Scene : register(t0, space200);
RWTexture2D<float4> RenderTarget : register(u0);
RWStructuredBuffer<PathState> PathStates : register(u1);
RWStructuredBuffer<RayWorkItem> RayQueueA : register(u2);
RWStructuredBuffer<RayWorkItem> RayQueueB : register(u3);
RWStructuredBuffer<ShadowWorkItem> ShadowQueue : register(u4);
RWStructuredBuffer<uint> WavefrontCounters : register(u5);
RWStructuredBuffer<HitWorkItem> HitQueueA : register(u6);
RWStructuredBuffer<HitWorkItem> HitQueueB : register(u7);
RWByteAddressBuffer WavefrontDispatchArgs : register(u8);

ConstantBuffer<RayTraceConstants> RayTraceCB : register(b0);
ConstantBuffer<LightConstants> LightCBuffer : register(b1);

SamplerState MeshSampler : register(s0);
SamplerState LinearSampler : register(s1);

static const uint Counter_CurrentRays = 0;
static const uint Counter_NextRays = 1;
static const uint Counter_Shadows = 2;
static const uint Counter_Hits = 3;
static const uint Counter_ReorderBinCounts = 4;
static const uint Counter_ReorderBinOffsets = 68;

static const uint PathFlag_Diffuse = 1u;
static const uint QueueA = 0u;
static const uint QueueB = 1u;
static const uint RayTraceFlag_EnableWavefrontReorder = 2u;
static const uint NumReorderBins = 64u;
static const uint DispatchArgs_CurrentRays = 0u;
static const uint DispatchArgs_Hits = 1u;
static const uint DispatchArgs_Shadows = 2u;
static const uint DispatchArgsStrideBytes = 12u;

static float2 SamplePoint(in uint pixelIdx, inout uint setIdx)
{
    const uint permutation = setIdx * RayTraceCB.TotalNumPixels + pixelIdx;
    setIdx += 1;
    return SampleCMJ2D(RayTraceCB.CurrSampleIdx, AppSettings.SqrtNumSamples, AppSettings.SqrtNumSamples, permutation);
}

static RayWorkItem LoadRayWorkItem(in uint queueIdx, in uint itemIdx)
{
    RayWorkItem item;
    if(queueIdx == QueueA)
        item = RayQueueA[itemIdx];
    else
        item = RayQueueB[itemIdx];
    return item;
}

static void StoreRayWorkItem(in uint queueIdx, in uint itemIdx, in RayWorkItem item)
{
    if(queueIdx == QueueA)
        RayQueueA[itemIdx] = item;
    else
        RayQueueB[itemIdx] = item;
}

static HitWorkItem LoadHitWorkItem(in uint queueIdx, in uint itemIdx)
{
    HitWorkItem item;
    if(queueIdx == QueueA)
        item = HitQueueA[itemIdx];
    else
        item = HitQueueB[itemIdx];
    return item;
}

static void StoreHitWorkItem(in uint queueIdx, in uint itemIdx, in HitWorkItem item)
{
    if(queueIdx == QueueA)
        HitQueueA[itemIdx] = item;
    else
        HitQueueB[itemIdx] = item;
}

static uint WavefrontHitSortKey(in HitWorkItem hitItem)
{
    const uint key = hitItem.GeometryIdx | (hitItem.PrimitiveIdx << 16);
    return key & (NumReorderBins - 1);
}

MeshVertex GetHitSurface_RQ(in float2 bary2, in uint geometryIdx, in uint primitiveIdx)
{
    float3 barycentrics = float3(1.0f - bary2.x - bary2.y, bary2.x, bary2.y);

    StructuredBuffer<GeometryInfo> geoInfoBuffer = ResourceDescriptorHeap[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[geometryIdx];

    StructuredBuffer<MeshVertex> vtxBuffer = ResourceDescriptorHeap[RayTraceCB.VtxBufferIdx];
    Buffer<uint> idxBuffer = ResourceDescriptorHeap[RayTraceCB.IdxBufferIdx];

    const uint idx0 = idxBuffer[primitiveIdx * 3 + geoInfo.IdxOffset + 0];
    const uint idx1 = idxBuffer[primitiveIdx * 3 + geoInfo.IdxOffset + 1];
    const uint idx2 = idxBuffer[primitiveIdx * 3 + geoInfo.IdxOffset + 2];

    const MeshVertex vtx0 = vtxBuffer[idx0 + geoInfo.VtxOffset];
    const MeshVertex vtx1 = vtxBuffer[idx1 + geoInfo.VtxOffset];
    const MeshVertex vtx2 = vtxBuffer[idx2 + geoInfo.VtxOffset];

    return BarycentricLerp(vtx0, vtx1, vtx2, barycentrics);
}

Material GetGeometryMaterial_RQ(in uint geometryIdx)
{
    StructuredBuffer<GeometryInfo> geoInfoBuffer = ResourceDescriptorHeap[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[geometryIdx];

    StructuredBuffer<Material> materialBuffer = ResourceDescriptorHeap[RayTraceCB.MaterialBufferIdx];
    return materialBuffer[geoInfo.MaterialIdx];
}

static bool PassAlphaTest_RQ(in uint geometryIdx, in uint primitiveIdx, in float2 bary2)
{
    const MeshVertex hitSurface = GetHitSurface_RQ(bary2, geometryIdx, primitiveIdx);
    const Material material = GetGeometryMaterial_RQ(geometryIdx);

    Texture2D opacityMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Opacity)];
    return opacityMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x >= 0.35f;
}

static HitInfoRQ TraceClosestHitInline_Radiance(in RayDesc ray, in uint rayMask, in uint rayFlags)
{
    HitInfoRQ outHit;
    outHit.Hit = false;
    outHit.T = 0.0f;
    outHit.GeometryIdx = 0;
    outHit.PrimitiveIdx = 0;
    outHit.Bary = 0.0.xx;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(Scene, rayFlags, rayMask, ray);

    while(q.Proceed())
    {
        if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            if(PassAlphaTest_RQ(q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
                q.CommitNonOpaqueTriangleHit();
        }
    }

    if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        outHit.Hit = true;
        outHit.T = q.CommittedRayT();
        outHit.GeometryIdx = q.CommittedGeometryIndex();
        outHit.PrimitiveIdx = q.CommittedPrimitiveIndex();
        outHit.Bary = q.CommittedTriangleBarycentrics();
    }

    return outHit;
}

static float TraceShadowInline(in RayDesc ray, in uint rayMask, in uint rayFlags)
{
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES> q;
    q.TraceRayInline(Scene, rayFlags, rayMask, ray);

    while(q.Proceed())
    {
        if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            if(PassAlphaTest_RQ(q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
                q.CommitNonOpaqueTriangleHit();
        }
    }

    return q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 0.0f : 1.0f;
}

static float3 EvaluateMissRadiance(in float3 rayDirWS, in uint pathLength)
{
    if(AppSettings.EnableWhiteFurnaceMode)
        return 1.0.xxx;

    TextureCube skyTexture = ResourceDescriptorHeap[RayTraceCB.SkyTextureIdx];
    float3 radiance = AppSettings.EnableSky ? skyTexture.SampleLevel(LinearSampler, rayDirWS, 0.0f).xyz : 0.0.xxx;

    if(pathLength == 1)
    {
        float cosSunAngle = dot(rayDirWS, RayTraceCB.SunDirectionWS);
        if(cosSunAngle >= RayTraceCB.CosSunAngularRadius)
            radiance = RayTraceCB.SunRenderColor;
    }

    return radiance;
}

static bool ShadeSurfaceAndSampleNext(
    in MeshVertex hitSurface,
    in Material material,
    in float3 incomingRayOriginWS,
    in float3 incomingRayDirWS,
    in uint pixelIdx,
    inout uint sampleSetIdx,
    in uint pathLength,
    in bool inIsDiffuse,
    in float inClampRoughnessValue,
    out float3 outRadianceAdd,
    out float3 outSunContribution,
    out ShadowWorkItem outSunShadow,
    out NextBounce outNext)
{
    outRadianceAdd = 0.0.xxx;
    outSunContribution = 0.0.xxx;
    outSunShadow.Origin = 0.0.xxx;
    outSunShadow.TMin = 0.0f;
    outSunShadow.Direction = 0.0.xxx;
    outSunShadow.TMax = 0.0f;
    outSunShadow.Contribution = 0.0.xxx;
    outSunShadow.PathStateIdx = 0;
    outNext.Valid = false;
    outNext.DirWS = 0.0.xxx;
    outNext.Throughput = 0.0.xxx;
    outNext.Roughness = 0.0f;
    outNext.IsDiffuse = false;

    if((!AppSettings.EnableDiffuse && !AppSettings.EnableSpecular) ||
       (!AppSettings.EnableDirect && !AppSettings.EnableIndirect))
        return false;

    if(pathLength > 1 && !AppSettings.EnableIndirect)
        return false;

    float3x3 tangentToWorld = float3x3(hitSurface.Tangent, hitSurface.Bitangent, hitSurface.Normal);
    const float3 positionWS = hitSurface.Position;

    float3 normalWS = hitSurface.Normal;
    if(AppSettings.EnableNormalMaps)
    {
        Texture2D normalMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Normal)];

        float3 normalTS;
        normalTS.xy = normalMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xy * 2.0f - 1.0f;
        normalTS.z = sqrt(1.0f - saturate(normalTS.x * normalTS.x + normalTS.y * normalTS.y));
        normalWS = normalize(mul(normalTS, tangentToWorld));

        tangentToWorld._31_32_33 = normalWS;
    }

    float3 baseColor = 1.0f;
    if(AppSettings.EnableAlbedoMaps && !AppSettings.EnableWhiteFurnaceMode)
    {
        Texture2D albedoMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Albedo)];
        baseColor = albedoMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;
    }

    Texture2D metallicMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Metallic)];
    const float metallic = saturate((AppSettings.EnableWhiteFurnaceMode ? 1.0f : metallicMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x) * AppSettings.MetallicScale);

    const bool enableDiffuse = (AppSettings.EnableDiffuse && metallic < 1.0f) || AppSettings.EnableWhiteFurnaceMode;
    const bool enableSpecular = (AppSettings.EnableSpecular && (AppSettings.EnableIndirectSpecular ? !(AppSettings.AvoidCausticPaths && inIsDiffuse) : (pathLength == 1)));

    if(enableDiffuse == false && enableSpecular == false)
        return false;

    Texture2D roughnessMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Roughness)];
    const float sqrtRoughness = saturate((AppSettings.EnableWhiteFurnaceMode ? 1.0f : roughnessMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x) * AppSettings.RoughnessScale);

    const float3 diffuseAlbedo = lerp(baseColor, 0.0f, metallic) * (enableDiffuse ? 1.0f : 0.0f);
    const float3 specularAlbedo = lerp(0.03f, baseColor, metallic) * (enableSpecular ? 1.0f : 0.0f);

    float roughness = sqrtRoughness * sqrtRoughness;
    if(AppSettings.ClampRoughness)
        roughness = max(roughness, inClampRoughnessValue);

    float3 msEnergyCompensation = 1.0.xxx;
    if(AppSettings.ApplyMultiscatteringEnergyCompensation)
    {
        float2 DFG = GGXEnvironmentBRDFScaleBias(saturate(dot(normalWS, -incomingRayDirWS)), sqrtRoughness);
        float Ess = DFG.x;
        msEnergyCompensation = 1.0.xxx + specularAlbedo * (1.0f / Ess - 1.0f);
    }

    Texture2D emissiveMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Emissive)];
    float3 radiance = AppSettings.EnableWhiteFurnaceMode ? 0.0.xxx : emissiveMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;

    if(AppSettings.EnableSun && !AppSettings.EnableWhiteFurnaceMode)
    {
        float3 sunDirection = RayTraceCB.SunDirectionWS;

        if(AppSettings.SunAreaLightApproximation)
        {
            float3 D = RayTraceCB.SunDirectionWS;
            float3 R = reflect(incomingRayDirWS, normalWS);
            float r = RayTraceCB.SinSunAngularRadius;
            float d = RayTraceCB.CosSunAngularRadius;
            float DDotR = dot(D, R);
            float3 S = R - DDotR * D;
            sunDirection = DDotR < d ? normalize(d * D + normalize(S) * r) : R;
        }

        outSunShadow.Origin = positionWS;
        outSunShadow.TMin = 0.00001f;
        outSunShadow.Direction = RayTraceCB.SunDirectionWS;
        outSunShadow.TMax = FP32Max;
        outSunContribution = CalcLighting(normalWS, sunDirection, RayTraceCB.SunIrradiance, diffuseAlbedo, specularAlbedo,
                                          roughness, positionWS, incomingRayOriginWS, msEnergyCompensation);
    }

    if(AppSettings.RenderLights)
    {
        for(uint spotLightIdx = 0; spotLightIdx < RayTraceCB.NumLights; spotLightIdx++)
        {
            SpotLight spotLight = LightCBuffer.Lights[spotLightIdx];

            float3 surfaceToLight = spotLight.Position - positionWS;
            float distanceToLight = length(surfaceToLight);
            surfaceToLight /= distanceToLight;
            float angleFactor = saturate(dot(surfaceToLight, spotLight.Direction));
            float angularAttenuation = smoothstep(spotLight.AngularAttenuationY, spotLight.AngularAttenuationX, angleFactor);

            float d = distanceToLight / spotLight.Range;
            float falloff = saturate(1.0f - (d * d * d * d));
            falloff = (falloff * falloff) / (distanceToLight * distanceToLight + 1.0f);

            angularAttenuation *= falloff;

            if(angularAttenuation > 0.0f)
            {
                RayDesc shadowRay;
                shadowRay.Origin = positionWS + normalWS * 0.01f;
                shadowRay.Direction = surfaceToLight;
                shadowRay.TMin = SpotShadowNearClip;
                shadowRay.TMax = distanceToLight - SpotShadowNearClip;

                uint shadowFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
                if(pathLength > AppSettings.MaxAnyHitPathLength)
                    shadowFlags |= RAY_FLAG_FORCE_OPAQUE;

                const float visibility = TraceShadowInline(shadowRay, 0xFFFFFFFF, shadowFlags);
                float3 intensity = spotLight.Intensity * angularAttenuation;

                radiance += CalcLighting(normalWS, surfaceToLight, intensity, diffuseAlbedo, specularAlbedo,
                                         roughness, positionWS, incomingRayOriginWS, msEnergyCompensation) * visibility;
            }
        }
    }

    if(pathLength == 1 && !AppSettings.EnableDirect)
    {
        radiance = 0.0.xxx;
        outSunContribution = 0.0.xxx;
    }

    float2 brdfSample = SamplePoint(pixelIdx, sampleSetIdx);

    float3 throughput = 0.0.xxx;
    float3 rayDirTS = 0.0.xxx;

    float selector = brdfSample.x;
    if(enableSpecular == false)
        selector = 0.0f;
    else if(enableDiffuse == false)
        selector = 1.0f;

    bool nextIsDiffuse = false;

    if(selector < 0.5f)
    {
        if(enableSpecular)
            brdfSample.x *= 2.0f;

        rayDirTS = SampleDirectionCosineHemisphere(brdfSample.x, brdfSample.y);
        throughput = diffuseAlbedo;
        nextIsDiffuse = true;
    }
    else
    {
        if(enableDiffuse)
            brdfSample.x = (brdfSample.x - 0.5f) * 2.0f;

        float3 incomingRayDirTS = normalize(mul(incomingRayDirWS, transpose(tangentToWorld)));
        float3 microfacetNormalTS = SampleGGXVisibleNormal(-incomingRayDirTS, roughness, roughness, brdfSample.x, brdfSample.y);
        float3 sampleDirTS = reflect(incomingRayDirTS, microfacetNormalTS);

        float3 normalTS = float3(0.0f, 0.0f, 1.0f);

        float3 F = AppSettings.EnableWhiteFurnaceMode ? 1.0.xxx : Fresnel(specularAlbedo, microfacetNormalTS, sampleDirTS);
        float G1 = SmithGGXMasking(normalTS, sampleDirTS, -incomingRayDirTS, roughness * roughness);
        float G2 = SmithGGXMaskingShadowing(normalTS, sampleDirTS, -incomingRayDirTS, roughness * roughness);

        throughput = F * (G2 / G1);
        rayDirTS = sampleDirTS;

        if(AppSettings.ApplyMultiscatteringEnergyCompensation)
        {
            float2 DFG = GGXEnvironmentBRDFScaleBias(saturate(dot(normalTS, -incomingRayDirWS)), sqrtRoughness);
            float Ess = DFG.x;
            throughput *= 1.0.xxx + specularAlbedo * (1.0f / Ess - 1.0f);
        }
    }

    const float3 rayDirWS = normalize(mul(rayDirTS, tangentToWorld));

    if(enableDiffuse && enableSpecular)
        throughput *= 2.0f;

    outRadianceAdd = radiance;
    outNext.Valid = true;
    outNext.DirWS = rayDirWS;
    outNext.Throughput = throughput;
    outNext.Roughness = roughness;
    outNext.IsDiffuse = nextIsDiffuse;
    return true;
}

[numthreads(64, 1, 1)]
void WavefrontClearCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if(dispatchThreadID.x < 4)
        WavefrontCounters[dispatchThreadID.x] = 0;
}

[numthreads(8, 8, 1)]
void WavefrontGeneratePrimaryCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int width;
    int height;
    RenderTarget.GetDimensions(width, height);

    const uint2 pixelCoord = dispatchThreadID.xy;
    if(pixelCoord.x >= uint(width) || pixelCoord.y >= uint(height))
        return;

    const uint pixelIdx = pixelCoord.y * uint(width) + pixelCoord.x;
    uint sampleSetIdx = 0;

    float2 primaryRaySample = SamplePoint(pixelIdx, sampleSetIdx);

    float2 rayPixelPos = pixelCoord + primaryRaySample;
    float2 ncdXY = (rayPixelPos / (float2(width, height) * 0.5f)) - 1.0f;
    ncdXY.y *= -1.0f;

    float4 rayStart = mul(float4(ncdXY, 0.0f, 1.0f), RayTraceCB.InvViewProjection);
    float4 rayEnd = mul(float4(ncdXY, 1.0f, 1.0f), RayTraceCB.InvViewProjection);

    rayStart.xyz /= rayStart.w;
    rayEnd.xyz /= rayEnd.w;

    RayWorkItem ray;
    ray.Origin = rayStart.xyz;
    ray.TMin = 0.0f;
    ray.Direction = normalize(rayEnd.xyz - rayStart.xyz);
    ray.TMax = length(rayEnd.xyz - rayStart.xyz);
    ray.PathStateIdx = pixelIdx;
    ray.Padding0 = 0;
    ray.Padding1 = 0;
    ray.Padding2 = 0;

    PathState state;
    state.Radiance = 0.0.xxx;
    state.Throughput = 1.0.xxx;
    state.PixelIdx = pixelIdx;
    state.SampleSetIdx = sampleSetIdx;
    state.PathLength = 1;
    state.Flags = 0;
    state.Roughness = 0.0f;
    state.Padding = 0.0f;

    PathStates[pixelIdx] = state;
    RayQueueA[pixelIdx] = ray;

    if(pixelIdx == 0)
        WavefrontCounters[Counter_CurrentRays] = RayTraceCB.TotalNumPixels;
}

[numthreads(64, 1, 1)]
void WavefrontTraceHitsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint workIdx = dispatchThreadID.x;
    if(workIdx >= WavefrontCounters[Counter_CurrentRays])
        return;

    RayWorkItem rayItem = LoadRayWorkItem(RayTraceCB.WavefrontReadQueue, workIdx);
    PathState state = PathStates[rayItem.PathStateIdx];

    RayDesc ray;
    ray.Origin = rayItem.Origin;
    ray.Direction = rayItem.Direction;
    ray.TMin = rayItem.TMin;
    ray.TMax = rayItem.TMax;

    uint rayFlags = 0;
    if(state.PathLength > AppSettings.MaxAnyHitPathLength)
        rayFlags |= RAY_FLAG_FORCE_OPAQUE;

    HitInfoRQ hit = TraceClosestHitInline_Radiance(ray, 0xFFFFFFFF, rayFlags);

    if(hit.Hit == false)
    {
        state.Radiance += state.Throughput * EvaluateMissRadiance(ray.Direction, state.PathLength);
        PathStates[rayItem.PathStateIdx] = state;
        return;
    }

    HitWorkItem hitItem;
    hitItem.PathStateIdx = rayItem.PathStateIdx;
    hitItem.GeometryIdx = hit.GeometryIdx;
    hitItem.PrimitiveIdx = hit.PrimitiveIdx;
    hitItem.Bary = hit.Bary;
    hitItem.RayOrigin = ray.Origin;
    hitItem.RayT = hit.T;
    hitItem.RayDirection = ray.Direction;
    hitItem.Padding = 0;

    uint hitIdx = 0;
    InterlockedAdd(WavefrontCounters[Counter_Hits], 1, hitIdx);
    StoreHitWorkItem(RayTraceCB.WavefrontReadQueue, hitIdx, hitItem);
}

[numthreads(1, 1, 1)]
void WavefrontPrepareDispatchArgsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint currentRayGroups = (WavefrontCounters[Counter_CurrentRays] + 63u) / 64u;
    const uint hitGroups = (WavefrontCounters[Counter_Hits] + 63u) / 64u;
    const uint shadowGroups = (WavefrontCounters[Counter_Shadows] + 63u) / 64u;

    const uint currentRayOffset = DispatchArgs_CurrentRays * DispatchArgsStrideBytes;
    WavefrontDispatchArgs.Store(currentRayOffset + 0u, currentRayGroups);
    WavefrontDispatchArgs.Store(currentRayOffset + 4u, 1u);
    WavefrontDispatchArgs.Store(currentRayOffset + 8u, 1u);

    const uint hitOffset = DispatchArgs_Hits * DispatchArgsStrideBytes;
    WavefrontDispatchArgs.Store(hitOffset + 0u, hitGroups);
    WavefrontDispatchArgs.Store(hitOffset + 4u, 1u);
    WavefrontDispatchArgs.Store(hitOffset + 8u, 1u);

    const uint shadowOffset = DispatchArgs_Shadows * DispatchArgsStrideBytes;
    WavefrontDispatchArgs.Store(shadowOffset + 0u, shadowGroups);
    WavefrontDispatchArgs.Store(shadowOffset + 4u, 1u);
    WavefrontDispatchArgs.Store(shadowOffset + 8u, 1u);
}

[numthreads(64, 1, 1)]
void WavefrontShadeHitsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint workIdx = dispatchThreadID.x;
    if(workIdx >= WavefrontCounters[Counter_Hits])
        return;

    HitWorkItem hitItem = LoadHitWorkItem(RayTraceCB.WavefrontReadQueue, workIdx);
    PathState state = PathStates[hitItem.PathStateIdx];

    MeshVertex hitSurface = GetHitSurface_RQ(hitItem.Bary, hitItem.GeometryIdx, hitItem.PrimitiveIdx);
    Material material = GetGeometryMaterial_RQ(hitItem.GeometryIdx);

    float3 addRadiance;
    float3 sunContribution;
    ShadowWorkItem sunShadow;
    NextBounce next;
    bool shaded = ShadeSurfaceAndSampleNext(hitSurface, material, hitItem.RayOrigin, hitItem.RayDirection, state.PixelIdx,
                                            state.SampleSetIdx, state.PathLength, (state.Flags & PathFlag_Diffuse) != 0,
                                            state.Roughness, addRadiance, sunContribution, sunShadow, next);

    if(shaded == false)
    {
        PathStates[hitItem.PathStateIdx] = state;
        return;
    }

    state.Radiance += state.Throughput * addRadiance;

    if(any(sunContribution != 0.0.xxx))
    {
        uint shadowIdx = 0;
        InterlockedAdd(WavefrontCounters[Counter_Shadows], 1, shadowIdx);
        sunShadow.Contribution = state.Throughput * sunContribution;
        sunShadow.PathStateIdx = hitItem.PathStateIdx;
        ShadowQueue[shadowIdx] = sunShadow;
    }

    if(next.Valid == false)
    {
        PathStates[hitItem.PathStateIdx] = state;
        return;
    }

    const bool canContinue = AppSettings.EnableIndirect &&
                             ((state.PathLength + 1) < AppSettings.MaxPathLength) &&
                             (!AppSettings.EnableWhiteFurnaceMode);

    if(canContinue == false)
    {
        RayDesc visRay;
        visRay.Origin = hitSurface.Position;
        visRay.Direction = next.DirWS;
        visRay.TMin = 0.00001f;
        visRay.TMax = FP32Max;

        uint visFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
        if((state.PathLength + 1) > AppSettings.MaxAnyHitPathLength)
            visFlags |= RAY_FLAG_FORCE_OPAQUE;

        if(AppSettings.EnableWhiteFurnaceMode)
        {
            state.Radiance += state.Throughput * next.Throughput;
        }
        else
        {
            const float visibility = TraceShadowInline(visRay, 0xFFFFFFFF, visFlags);
            TextureCube skyTexture = ResourceDescriptorHeap[RayTraceCB.SkyTextureIdx];
            float3 skyRadiance = AppSettings.EnableSky ? skyTexture.SampleLevel(LinearSampler, next.DirWS, 0.0f).xyz : 0.0.xxx;
            state.Radiance += state.Throughput * visibility * skyRadiance * next.Throughput;
        }

        PathStates[hitItem.PathStateIdx] = state;
        return;
    }

    state.Throughput *= next.Throughput;
    state.PathLength += 1;
    state.Roughness = next.Roughness;
    state.Flags = next.IsDiffuse ? PathFlag_Diffuse : 0;

    RayWorkItem nextRay;
    nextRay.Origin = hitSurface.Position;
    nextRay.TMin = 0.00001f;
    nextRay.Direction = next.DirWS;
    nextRay.TMax = FP32Max;
    nextRay.PathStateIdx = hitItem.PathStateIdx;
    nextRay.Padding0 = 0;
    nextRay.Padding1 = 0;
    nextRay.Padding2 = 0;

    uint nextIdx = 0;
    InterlockedAdd(WavefrontCounters[Counter_NextRays], 1, nextIdx);
    StoreRayWorkItem(RayTraceCB.WavefrontWriteQueue, nextIdx, nextRay);
    PathStates[hitItem.PathStateIdx] = state;
}

[numthreads(64, 1, 1)]
void WavefrontClearReorderCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint bin = dispatchThreadID.x;
    if(bin < NumReorderBins)
    {
        WavefrontCounters[Counter_ReorderBinCounts + bin] = 0;
        WavefrontCounters[Counter_ReorderBinOffsets + bin] = 0;
    }
}

[numthreads(64, 1, 1)]
void WavefrontCountReorderBinsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint workIdx = dispatchThreadID.x;
    if(workIdx >= WavefrontCounters[Counter_Hits])
        return;

    HitWorkItem hitItem = LoadHitWorkItem(RayTraceCB.WavefrontReadQueue, workIdx);
    const uint bin = WavefrontHitSortKey(hitItem);
    InterlockedAdd(WavefrontCounters[Counter_ReorderBinCounts + bin], 1);
}

[numthreads(1, 1, 1)]
void WavefrontPrefixReorderBinsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint offset = 0;
    for(uint bin = 0; bin < NumReorderBins; ++bin)
    {
        const uint count = WavefrontCounters[Counter_ReorderBinCounts + bin];
        WavefrontCounters[Counter_ReorderBinOffsets + bin] = offset;
        WavefrontCounters[Counter_ReorderBinCounts + bin] = 0;
        offset += count;
    }
}

[numthreads(64, 1, 1)]
void WavefrontScatterReorderedRaysCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint workIdx = dispatchThreadID.x;
    if(workIdx >= WavefrontCounters[Counter_Hits])
        return;

    HitWorkItem hitItem = LoadHitWorkItem(RayTraceCB.WavefrontReadQueue, workIdx);
    const uint bin = WavefrontHitSortKey(hitItem);

    uint localIdx = 0;
    InterlockedAdd(WavefrontCounters[Counter_ReorderBinCounts + bin], 1, localIdx);
    StoreHitWorkItem(RayTraceCB.WavefrontWriteQueue, WavefrontCounters[Counter_ReorderBinOffsets + bin] + localIdx, hitItem);
}

[numthreads(64, 1, 1)]
void WavefrontTraceShadowsCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint workIdx = dispatchThreadID.x;
    if(workIdx >= WavefrontCounters[Counter_Shadows])
        return;

    ShadowWorkItem shadow = ShadowQueue[workIdx];
    PathState state = PathStates[shadow.PathStateIdx];

    RayDesc ray;
    ray.Origin = shadow.Origin;
    ray.Direction = shadow.Direction;
    ray.TMin = shadow.TMin;
    ray.TMax = shadow.TMax;

    uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    if(state.PathLength > AppSettings.MaxAnyHitPathLength)
        rayFlags |= RAY_FLAG_FORCE_OPAQUE;

    const float visibility = TraceShadowInline(ray, 0xFFFFFFFF, rayFlags);
    state.Radiance += visibility * shadow.Contribution;
    PathStates[shadow.PathStateIdx] = state;
}

[numthreads(64, 1, 1)]
void WavefrontAdvanceCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    if(dispatchThreadID.x == 0)
    {
        WavefrontCounters[Counter_CurrentRays] = WavefrontCounters[Counter_NextRays];
        WavefrontCounters[Counter_NextRays] = 0;
        WavefrontCounters[Counter_Shadows] = 0;
        WavefrontCounters[Counter_Hits] = 0;
    }
}

[numthreads(8, 8, 1)]
void WavefrontAccumulateCS(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int width;
    int height;
    RenderTarget.GetDimensions(width, height);

    const uint2 pixelCoord = dispatchThreadID.xy;
    if(pixelCoord.x >= uint(width) || pixelCoord.y >= uint(height))
        return;

    const uint pixelIdx = pixelCoord.y * uint(width) + pixelCoord.x;
    float3 newSample = clamp(PathStates[pixelIdx].Radiance, 0.0f, FP16Max);

    const float lerpFactor = RayTraceCB.CurrSampleIdx / (RayTraceCB.CurrSampleIdx + 1.0f);
    float3 currValue = RenderTarget[pixelCoord].xyz;
    float3 newValue = lerp(newSample, currValue, lerpFactor);

    if(RayTraceCB.CurrSampleIdx == 0)
        newValue = newSample;

    RenderTarget[pixelCoord] = float4(newValue, 1.0f);
}
