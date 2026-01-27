//=================================================================================================
//
//  Modified to make use of DXR 1.1 (RayQuery). Template-based recursion flavor.
//
//  DXR Path Tracer
//  by MJP
//  http://mynameismjp.wordpress.com/
//
//  All code and content licensed under the MIT license
//
//=================================================================================================

//=================================================================================================
// Includes
//=================================================================================================
#include <DescriptorTables.hlsl>
#include <Constants.hlsl>
#include <Quaternion.hlsl>
#include <BRDF.hlsl>
#include <RayTracing.hlsl>
#include <Sampling.hlsl>

#include "SharedTypes.h"
#include "AppSettings.hlsl"

#define RECURSION_LIMIT 7

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
    uint IdxBufferIdx;
    uint GeometryInfoBufferIdx;
    uint MaterialBufferIdx;
    uint SkyTextureIdx;
    uint NumLights;
};

struct LightConstants
{
    SpotLight Lights[MaxSpotLights];
    float4x4 ShadowMatrices[MaxSpotLights];
};

RaytracingAccelerationStructure Scene : register(t0, space200);
RWTexture2D<float4> RenderTarget : register(u0);

ConstantBuffer<RayTraceConstants> RayTraceCB : register(b0);

ConstantBuffer<LightConstants> LightCBuffer : register(b1);

SamplerState MeshSampler : register(s0);
SamplerState LinearSampler : register(s1);

typedef BuiltInTriangleIntersectionAttributes HitAttributes;
struct PrimaryPayload
{
    float3 Radiance;
    float Roughness;
    uint PathLength;
    uint PixelIdx;
    uint SampleSetIdx;
    bool IsDiffuse;
};

struct ShadowPayload
{
    float Visibility;
};

enum RayTypes {
    RayTypeRadiance = 0,
    RayTypeShadow = 1,

    NumRayTypes
};

// Loops up the vertex data for the hit triangle and interpolates its attributes
MeshVertex GetHitSurface(in HitAttributes attr, in uint geometryIdx, in uint primIdx)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    StructuredBuffer<GeometryInfo> geoInfoBuffer = ResourceDescriptorHeap[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[geometryIdx];

    StructuredBuffer<MeshVertex> vtxBuffer = ResourceDescriptorHeap[RayTraceCB.VtxBufferIdx];
    Buffer<uint> idxBuffer = ResourceDescriptorHeap[RayTraceCB.IdxBufferIdx];

    const uint idx0 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 0];
    const uint idx1 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 1];
    const uint idx2 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 2];

    const MeshVertex vtx0 = vtxBuffer[idx0 + geoInfo.VtxOffset];
    const MeshVertex vtx1 = vtxBuffer[idx1 + geoInfo.VtxOffset];
    const MeshVertex vtx2 = vtxBuffer[idx2 + geoInfo.VtxOffset];

    return BarycentricLerp(vtx0, vtx1, vtx2, barycentrics);
}

// Gets the material assigned to a geometry in the acceleration structure
Material GetGeometryMaterial(in uint geometryIdx)
{
    StructuredBuffer<GeometryInfo> geoInfoBuffer = ResourceDescriptorHeap[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[geometryIdx];

    StructuredBuffer<Material> materialBuffer = ResourceDescriptorHeap[RayTraceCB.MaterialBufferIdx];
    return materialBuffer[geoInfo.MaterialIdx];
}

static float2 SamplePoint(in uint pixelIdx, inout uint setIdx)
{
    const uint permutation = setIdx * RayTraceCB.TotalNumPixels + pixelIdx;
    setIdx += 1;
    return SampleCMJ2D(RayTraceCB.CurrSampleIdx, AppSettings.SqrtNumSamples, AppSettings.SqrtNumSamples, permutation);
}

void MissShader(in float3 world_rd, inout PrimaryPayload payload)
{
    if (AppSettings.EnableWhiteFurnaceMode)
    {
        payload.Radiance = 1.0.xxx;
    }
    else
    {
        const float3 rayDir = world_rd;

        TextureCube skyTexture = ResourceDescriptorHeap[RayTraceCB.SkyTextureIdx];
        payload.Radiance = AppSettings.EnableSky ? skyTexture.SampleLevel(LinearSampler, rayDir, 0.0f).xyz : 0.0.xxx;

        if (payload.PathLength == 1)
        {
            float cosSunAngle = dot(rayDir, RayTraceCB.SunDirectionWS);
            if (cosSunAngle >= RayTraceCB.CosSunAngularRadius)
                payload.Radiance = RayTraceCB.SunRenderColor;
        }
    }
}

void ShadowMissShader(inout ShadowPayload payload)
{
    payload.Visibility = 1.0f;
}

template<uint Depth>
void MyTracePrimaryRay(uint traceRayFlags, uint instanceInclFlags,
    uint hitGroupOffset,
    uint hitGroupGeoMultiplier,
    uint missShaderIdx,
    RayDesc ray, inout PrimaryPayload payload);

void MyTraceShadowRay(uint traceRayFlags, uint instanceInclFlags,
    uint hitGroupOffset,
    uint hitGroupGeoMultiplier,
    uint missShaderIdx,
    RayDesc ray, inout ShadowPayload payload);

template<uint Depth>
float3 PathTrace(in MeshVertex hitSurface, in Material material, in PrimaryPayload inPayload, float3 world_ro, float3 world_rd)
{
    if ((!AppSettings.EnableDiffuse && !AppSettings.EnableSpecular) ||
        (!AppSettings.EnableDirect && !AppSettings.EnableIndirect))
        return 0.0.xxx;

    if (inPayload.PathLength > 1 && !AppSettings.EnableIndirect)
        return 0.0.xxx;

    float3x3 tangentToWorld = float3x3(hitSurface.Tangent, hitSurface.Bitangent, hitSurface.Normal);

    const float3 positionWS = hitSurface.Position;

    const float3 incomingRayOriginWS = world_ro;
    const float3 incomingRayDirWS = world_rd;

    float3 normalWS = hitSurface.Normal;
    if (AppSettings.EnableNormalMaps)
    {
        // Sample the normal map, and convert the normal to world space
        Texture2D normalMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Normal)];

        float3 normalTS;
        normalTS.xy = normalMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xy * 2.0f - 1.0f;
        normalTS.z = sqrt(1.0f - saturate(normalTS.x * normalTS.x + normalTS.y * normalTS.y));
        normalWS = normalize(mul(normalTS, tangentToWorld));

        tangentToWorld._31_32_33 = normalWS;
    }

    float3 baseColor = 1.0f;
    if (AppSettings.EnableAlbedoMaps && !AppSettings.EnableWhiteFurnaceMode)
    {
        Texture2D albedoMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Albedo)];
        baseColor = albedoMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;
    }

    Texture2D metallicMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Metallic)];
    const float metallic = saturate((AppSettings.EnableWhiteFurnaceMode ? 1.0f : metallicMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x) * AppSettings.MetallicScale);

    const bool enableDiffuse = (AppSettings.EnableDiffuse && metallic < 1.0f) || AppSettings.EnableWhiteFurnaceMode;
    const bool enableSpecular = (AppSettings.EnableSpecular && (AppSettings.EnableIndirectSpecular ? !(AppSettings.AvoidCausticPaths && inPayload.IsDiffuse) : (inPayload.PathLength == 1)));

    if (enableDiffuse == false && enableSpecular == false)
        return 0.0f;

    Texture2D roughnessMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Roughness)];
    const float sqrtRoughness = saturate((AppSettings.EnableWhiteFurnaceMode ? 1.0f : roughnessMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x) * AppSettings.RoughnessScale);

    const float3 diffuseAlbedo = lerp(baseColor, 0.0f, metallic) * (enableDiffuse ? 1.0f : 0.0f);
    const float3 specularAlbedo = lerp(0.03f, baseColor, metallic) * (enableSpecular ? 1.0f : 0.0f);
    float roughness = sqrtRoughness * sqrtRoughness;
    if (AppSettings.ClampRoughness)
        roughness = max(roughness, inPayload.Roughness);

    float3 msEnergyCompensation = 1.0.xxx;
    if (AppSettings.ApplyMultiscatteringEnergyCompensation)
    {
        float2 DFG = GGXEnvironmentBRDFScaleBias(saturate(dot(normalWS, -incomingRayDirWS)), sqrtRoughness);

        // Improve energy preservation by applying a scaled version of the original
        // single scattering specular lobe. Based on "Practical multiple scattering
        // compensation for microfacet models" [Turquin19].
        //
        // See: https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
        float Ess = DFG.x;
        msEnergyCompensation = 1.0.xxx + specularAlbedo * (1.0f / Ess - 1.0f);
    }

    Texture2D emissiveMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Emissive)];
    float3 radiance = AppSettings.EnableWhiteFurnaceMode ? 0.0.xxx : emissiveMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;

    //Apply sun light
    if (AppSettings.EnableSun && !AppSettings.EnableWhiteFurnaceMode)
    {
        float3 sunDirection = RayTraceCB.SunDirectionWS;

        if (AppSettings.SunAreaLightApproximation)
        {
            float3 D = RayTraceCB.SunDirectionWS;
            float3 R = reflect(incomingRayDirWS, normalWS);
            float r = RayTraceCB.SinSunAngularRadius;
            float d = RayTraceCB.CosSunAngularRadius;
            float DDotR = dot(D, R);
            float3 S = R - DDotR * D;
            sunDirection = DDotR < d ? normalize(d * D + normalize(S) * r) : R;
        }

        // Shoot a shadow ray to see if the sun is occluded
        RayDesc ray;
        ray.Origin = positionWS;
        ray.Direction = RayTraceCB.SunDirectionWS;
        ray.TMin = 0.00001f;
        ray.TMax = FP32Max;

        ShadowPayload payload;
        payload.Visibility = 1.0f;

        uint traceRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

        // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
        if (inPayload.PathLength > AppSettings.MaxAnyHitPathLength)
            traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

        const uint hitGroupOffset = RayTypeShadow;
        const uint hitGroupGeoMultiplier = NumRayTypes;
        const uint missShaderIdx = RayTypeShadow;
        MyTraceShadowRay(traceRayFlags, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);

        radiance += CalcLighting(normalWS, sunDirection, RayTraceCB.SunIrradiance, diffuseAlbedo, specularAlbedo,
                                 roughness, positionWS, incomingRayOriginWS, msEnergyCompensation) * payload.Visibility;
    }
    
    // Apply spot lights
    if (AppSettings.RenderLights)
    {
        //iterate all lights
        for (uint spotLightIdx = 0; spotLightIdx < RayTraceCB.NumLights; spotLightIdx++)
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

            if (angularAttenuation > 0.0f)
            {
                // Shoot a shadow ray to see if the sun is occluded
                RayDesc ray;
                ray.Origin = positionWS + normalWS * 0.01f;
                ray.Direction = surfaceToLight;
                ray.TMin = SpotShadowNearClip;
                ray.TMax = distanceToLight - SpotShadowNearClip;

                ShadowPayload payload;
                payload.Visibility = 1.0f;

                uint traceRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

                // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
                if (inPayload.PathLength > AppSettings.MaxAnyHitPathLength)
                    traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

                const uint hitGroupOffset = RayTypeShadow;
                const uint hitGroupGeoMultiplier = NumRayTypes;
                const uint missShaderIdx = RayTypeShadow;
                MyTraceShadowRay(traceRayFlags, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);

                float3 intensity = spotLight.Intensity * angularAttenuation;

                radiance += CalcLighting(normalWS, surfaceToLight, intensity, diffuseAlbedo, specularAlbedo,
                                         roughness, positionWS, incomingRayOriginWS, msEnergyCompensation) * payload.Visibility;
            }
        }
    }
    
    // Choose our next path by importance sampling our BRDFs
    float2 brdfSample = SamplePoint(inPayload.PixelIdx, inPayload.SampleSetIdx);

    float3 throughput = 0.0f;
    float3 rayDirTS = 0.0f;

    float selector = brdfSample.x;
    if (enableSpecular == false)
        selector = 0.0f;
    else if (enableDiffuse == false)
        selector = 1.0f;

    if (selector < 0.5f)
    {
        // We're sampling the diffuse BRDF, so sample a cosine-weighted hemisphere
        if (enableSpecular)
            brdfSample.x *= 2.0f;
        rayDirTS = SampleDirectionCosineHemisphere(brdfSample.x, brdfSample.y);

        // The PDF of sampling a cosine hemisphere is NdotL / Pi, which cancels out those terms
        // from the diffuse BRDF and the irradiance integral
        throughput = diffuseAlbedo;
    }
    else
    {
        // We're sampling the GGX specular BRDF by sampling the distribution of visible normals. See this post
        // for more info: https://schuttejoe.github.io/post/ggximportancesamplingpart2/.
        // Also see: https://hal.inria.fr/hal-00996995v1/document and https://hal.archives-ouvertes.fr/hal-01509746/document
        if (enableDiffuse)
            brdfSample.x = (brdfSample.x - 0.5f) * 2.0f;

        float3 incomingRayDirTS = normalize(mul(incomingRayDirWS, transpose(tangentToWorld)));
        float3 microfacetNormalTS = SampleGGXVisibleNormal(-incomingRayDirTS, roughness, roughness, brdfSample.x, brdfSample.y);
        float3 sampleDirTS = reflect(incomingRayDirTS, microfacetNormalTS);

        float3 normalTS = float3(0.0f, 0.0f, 1.0f);

        float3 F = AppSettings.EnableWhiteFurnaceMode ? 1.0.xxx : Fresnel(specularAlbedo, microfacetNormalTS, sampleDirTS);
        float G1 = SmithGGXMasking(normalTS, sampleDirTS, -incomingRayDirTS, roughness * roughness);
        float G2 = SmithGGXMaskingShadowing(normalTS, sampleDirTS, -incomingRayDirTS, roughness * roughness);

        throughput = (F * (G2 / G1));
        rayDirTS = sampleDirTS;

        if (AppSettings.ApplyMultiscatteringEnergyCompensation)
        {
            float2 DFG = GGXEnvironmentBRDFScaleBias(saturate(dot(normalTS, -incomingRayDirWS)), sqrtRoughness);

            // Improve energy preservation by applying a scaled version of the original
            // single scattering specular lobe. Based on "Practical multiple scattering
            // compensation for microfacet models" [Turquin19].
            //
            // See: https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
            float Ess = DFG.x;
            throughput *= 1.0.xxx + specularAlbedo * (1.0f / Ess - 1.0f);
        }
    }

    const float3 rayDirWS = normalize(mul(rayDirTS, tangentToWorld));

    if (enableDiffuse && enableSpecular)
        throughput *= 2.0f;

    // Shoot another ray to get the next path
    RayDesc ray;
    ray.Origin = positionWS;
    ray.Direction = rayDirWS;
    ray.TMin = 0.00001f;
    ray.TMax = FP32Max;

    if (inPayload.PathLength == 1 && !AppSettings.EnableDirect)
        radiance = 0.0.xxx;
    
    if (AppSettings.EnableIndirect && (inPayload.PathLength + 1 < AppSettings.MaxPathLength) && !AppSettings.EnableWhiteFurnaceMode)
    {
        PrimaryPayload payload;
        payload.Radiance = 0.0f;
        payload.PathLength = inPayload.PathLength + 1;
        payload.PixelIdx = inPayload.PixelIdx;
        payload.SampleSetIdx = inPayload.SampleSetIdx;
        payload.IsDiffuse = (selector < 0.5f);
        payload.Roughness = roughness;

        uint traceRayFlags = 0;

        // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
        if (payload.PathLength > AppSettings.MaxAnyHitPathLength)
            traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

        const uint hitGroupOffset = RayTypeRadiance;
        const uint hitGroupGeoMultiplier = NumRayTypes;
        const uint missShaderIdx = RayTypeRadiance;
        MyTracePrimaryRay<Depth+1>(traceRayFlags, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);

        radiance += payload.Radiance * throughput;
    }
    else
    {
        ShadowPayload payload;
        payload.Visibility = 1.0f;

        uint traceRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

        // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
        if (inPayload.PathLength + 1 > AppSettings.MaxAnyHitPathLength)
            traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

        const uint hitGroupOffset = RayTypeShadow;
        const uint hitGroupGeoMultiplier = NumRayTypes;
        const uint missShaderIdx = RayTypeShadow;
        MyTraceShadowRay(traceRayFlags, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);

        if (AppSettings.EnableWhiteFurnaceMode)
        {
            radiance = throughput;
        }
        else
        {
            TextureCube skyTexture = TexCubeTable[RayTraceCB.SkyTextureIdx];
            float3 skyRadiance = AppSettings.EnableSky ? skyTexture.SampleLevel(LinearSampler, rayDirWS, 0.0f).xyz : 0.0.xxx;

            radiance += payload.Visibility * skyRadiance * throughput;
        }
    }
    
    return radiance;
}

template<>
float3 PathTrace<RECURSION_LIMIT>(in MeshVertex hitSurface, in Material material, in PrimaryPayload inPayload, float3 world_ro, float3 world_rd){
    return float3(0, 0, 0);
}

template<uint Depth>
void MyTracePrimaryRay(uint traceRayFlags, uint instanceInclFlags,
    uint hitGroupOffset,
    uint hitGroupGeoMultiplier,
    uint missShaderIdx,
    RayDesc ray, inout PrimaryPayload payload)
{
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(Scene, traceRayFlags, instanceInclFlags, ray);
    while (q.Proceed())
    {
        switch (q.CandidateType())
        {
            case CANDIDATE_NON_OPAQUE_TRIANGLE:
                HitAttributes attr;
                attr.barycentrics = q.CandidateTriangleBarycentrics();
                const MeshVertex hitSurface = GetHitSurface(attr, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                const Material material = GetGeometryMaterial(q.CandidateGeometryIndex());

                // Standard alpha testing
                Texture2D opacityMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Opacity)];
                if (opacityMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x < 0.35f)
                {
                    // Ignore Hit
                }
                else
                {
                    q.CommitNonOpaqueTriangleHit();  // TODO
                }
                break;
        }
    }
    
    // Closest-hit Shader
    switch (q.CommittedStatus())
    {
        case COMMITTED_TRIANGLE_HIT:
        {
            HitAttributes attr;
            attr.barycentrics = q.CommittedTriangleBarycentrics();
            const MeshVertex hitSurface = GetHitSurface(attr, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
            const Material material = GetGeometryMaterial(q.CommittedGeometryIndex());
            payload.Radiance = PathTrace<Depth>(hitSurface, material, payload, q.WorldRayOrigin(), q.WorldRayDirection());
            break;
        }
        case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
        {
            break;
        }
        case COMMITTED_NOTHING:{
            MissShader(q.WorldRayDirection(), payload);
            break;
        }
    }
}

template<>
void MyTracePrimaryRay<RECURSION_LIMIT>(uint traceRayFlags, uint instanceInclFlags,
    uint hitGroupOffset,
    uint hitGroupGeoMultiplier,
    uint missShaderIdx,
    RayDesc ray, inout PrimaryPayload payload)
{
    
}

void MyTraceShadowRay(uint traceRayFlags, uint instanceInclFlags,
    uint hitGroupOffset,
    uint hitGroupGeoMultiplier,
    uint missShaderIdx,
    RayDesc ray, inout ShadowPayload payload)
{
    RayQuery < RAY_FLAG_NONE > q;
    q.TraceRayInline(Scene, traceRayFlags, instanceInclFlags, ray);
    while (q.Proceed())
    {
        switch (q.CandidateType())
        {
            case CANDIDATE_NON_OPAQUE_TRIANGLE:
                HitAttributes attr;
                attr.barycentrics = q.CandidateTriangleBarycentrics();
                const MeshVertex hitSurface = GetHitSurface(attr, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                const Material material = GetGeometryMaterial(q.CandidateGeometryIndex());

                // Standard alpha testing
                Texture2D opacityMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Opacity)];
                if (opacityMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x < 0.35f)
                {
                    // Ignore Hit
                }
                else
                {
                    q.CommitNonOpaqueTriangleHit(); // TODO
                }
                break;
        }
    }
    
    // Closest-hit Shader
    switch (q.CommittedStatus())
    {
        case COMMITTED_TRIANGLE_HIT:
        {
            payload.Visibility = 0.0f;
            break;
        }
        case COMMITTED_PROCEDURAL_PRIMITIVE_HIT:
        {
            break;
        }
        case COMMITTED_NOTHING:{
            payload.Visibility = 1.0f;
            break;
        }
    }
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int width;
    int height;
    RenderTarget.GetDimensions(width, height);
//    uint2 dixy = dispatchThreadID.xy;
//    RenderTarget[dixy] = float4(dixy / float2(width, height), 0, 1);
    
    const uint2 pixelCoord = dispatchThreadID.xy;
    const uint pixelIdx = pixelCoord.y * width + pixelCoord.x;
    
    uint sampleSetIdx = 0;
    
    // Form a primary ray by un-projecting the pixel coordinate using the inverse view * projection matrix
    float2 primaryRaySample = SamplePoint(pixelIdx, sampleSetIdx);
    
    float2 rayPixelPos = pixelCoord + primaryRaySample;
    float2 ncdXY = (rayPixelPos / (float2(width, height) * 0.5f)) - 1.0f;
    ncdXY.y *= -1.0f;
    float4 rayStart = mul(float4(ncdXY, 0.0f, 1.0f), RayTraceCB.InvViewProjection);
    float4 rayEnd = mul(float4(ncdXY, 1.0f, 1.0f), RayTraceCB.InvViewProjection);
    
    rayStart.xyz /= rayStart.w;
    rayEnd.xyz /= rayEnd.w;
    float3 rayDir = normalize(rayEnd.xyz - rayStart.xyz);
    float rayLength = length(rayEnd.xyz - rayStart.xyz);
    
    // Trace a primary ray
    RayDesc ray;
    ray.Origin = rayStart.xyz;
    ray.Direction = rayDir;
    ray.TMin = 0.0f;
    ray.TMax = rayLength;
    
    PrimaryPayload payload;
    payload.Radiance = 0.0f;
    payload.Roughness = 0.0f;
    payload.PathLength = 1;
    payload.PixelIdx = pixelIdx;
    payload.SampleSetIdx = sampleSetIdx;
    payload.IsDiffuse = false;

    uint traceRayFlags = 0;

    // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
    if (payload.PathLength > AppSettings.MaxAnyHitPathLength)
        traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

    const uint hitGroupOffset = RayTypeRadiance;
    const uint hitGroupGeoMultiplier = NumRayTypes;
    const uint missShaderIdx = RayTypeRadiance;
    
    MyTracePrimaryRay<0>(traceRayFlags, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);
    
    payload.Radiance = clamp(payload.Radiance, 0.0f, FP16Max);

    // Update the progressive result with the new radiance sample
    const float lerpFactor = RayTraceCB.CurrSampleIdx / (RayTraceCB.CurrSampleIdx + 1.0f);
    float3 newSample = payload.Radiance;
    float3 currValue = RenderTarget[pixelCoord].xyz;
    float3 newValue = lerp(newSample, currValue, lerpFactor);

    RenderTarget[pixelCoord] = float4(newValue, 1.0f);
}