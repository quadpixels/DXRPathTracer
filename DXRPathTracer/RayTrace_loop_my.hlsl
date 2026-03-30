//=================================================================================================
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

struct MyPayload
{
    uint gidx, pidx;
    float3 worldRayOrigin;
    float3 worldRayDirection;
    HitAttributes attr;
    float visibility;

    uint PathLength;  // Used in miss shader
    float3 radiance;  // Sky miss
};

struct MyStackEntry
{
    RayDesc ray;
    uint traceRayFlags;
    uint hitGroupOffset;
    uint hitGroupGeoMultiplier;
    uint missShaderIdx;
    uint step;
    uint resultSlot;  // Parent result
    uint prevStep;
    uint prevStepCkptIdx;
};

struct ClosestHitShaderCheckpoint {
    float3 radiance;
    uint step;  // PC
    
    uint shouldSendSunlight;
    uint shouldSendSpotlight;
    uint spotlightDone;
    uint spotlightIndex;
    
    uint pathLength;
    float3 normalWS;
    float3 sunDirection;
    float3 diffuseAlbedo;
    float3 specularAlbedo;
    float roughness;
    float sqrtRoughness;
    float3 positionWS;
    float3 incomingRayDirWS;
    float3 incomingRayOriginWS;
    float3 msEnergyCompensation;
    float angularAttenuation;
    float3 throughput;
    float3 rayDirTS;
    uint pixelIdx;
    uint sampleSetIdx;
    bool enableDiffuse;
    bool enableSpecular;
    float3x3 tangentToWorld;
    uint isDiffuse;
    
    bool hasRay;  // Do ray processing at the beginning of loop
    RayDesc nextRay;
    uint hitGroupOffset;
    uint hitGroupGeoMultiplier;
    uint missShaderIdx;
    uint traceRayFlags;
    
    float3 rayDirWS;
    bool hasNewCkpt;
    bool isChildRayRadiance;
};

enum RayTypes {
    RayTypeRadiance = 0,
    RayTypeShadow = 1,

    NumRayTypes
};

static float2 SamplePoint(in uint pixelIdx, inout uint setIdx)
{
    const uint permutation = setIdx * RayTraceCB.TotalNumPixels + pixelIdx;
    setIdx += 1;
    return SampleCMJ2D(RayTraceCB.CurrSampleIdx, AppSettings.SqrtNumSamples, AppSettings.SqrtNumSamples, permutation);
}

float3 IDtoColor(uint id)
{
    const float GOLDEN_RATIO = 0.61803398875f;

    float hue = frac(id * GOLDEN_RATIO);
    float saturation = 0.75f;
    float value = 0.95f;
    float3 hsv = float3(hue, saturation, value);

    float4 K = float4(1.f, 2.f / 3.f, 1.f / 3.f, 3.f);
    float3 p = abs(frac(hsv.xxx + K.xyz) * 6.f - K.www);
    return hsv.z * lerp(K.xxx, saturate(p - K.xxx), hsv.y);
}

float3 CalcLighting_my(in float3 normal, in float3 lightDir, in float3 peakIrradiance,
                    in float3 diffuseAlbedo, in float3 specularAlbedo, in float roughness,
                    in float3 positionWS, in float3 cameraPosWS, in float3 msEnergyCompensation)
{
    roughness = max(roughness, 0.01f);
    float3 lighting = diffuseAlbedo * (1.0f / 3.14159f);
    float3 view = normalize(cameraPosWS - positionWS);
    const float nDotL = saturate(dot(normal, lightDir));
    
    if(nDotL > 0.0f)
    {
        const float nDotV = saturate(dot(normal, view));
        float3 h = normalize(view + lightDir);

        float3 fresnel = Fresnel(specularAlbedo, h, lightDir);

        float specular = GGXSpecular(roughness, normal, h, view, lightDir);
        lighting += specular * fresnel * msEnergyCompensation;
    }
    return lighting * nDotL * peakIrradiance;
}

// Loops up the vertex data for the hit triangle and interpolates its attributes
MeshVertex GetHitSurface(in HitAttributes attr, in uint geometryIdx)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    StructuredBuffer<GeometryInfo> geoInfoBuffer = ResourceDescriptorHeap[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[geometryIdx];

    StructuredBuffer<MeshVertex> vtxBuffer = ResourceDescriptorHeap[RayTraceCB.VtxBufferIdx];
    Buffer<uint> idxBuffer = ResourceDescriptorHeap[RayTraceCB.IdxBufferIdx];

    const uint primIdx = PrimitiveIndex();
    const uint idx0 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 0];
    const uint idx1 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 1];
    const uint idx2 = idxBuffer[primIdx * 3 + geoInfo.IdxOffset + 2];

    const MeshVertex vtx0 = vtxBuffer[idx0 + geoInfo.VtxOffset];
    const MeshVertex vtx1 = vtxBuffer[idx1 + geoInfo.VtxOffset];
    const MeshVertex vtx2 = vtxBuffer[idx2 + geoInfo.VtxOffset];

    return BarycentricLerp(vtx0, vtx1, vtx2, barycentrics);
}

MeshVertex GetHitSurfaceMy(in HitAttributes attr, in uint geometryIdx, in uint primitiveIdx)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

    StructuredBuffer<GeometryInfo> geoInfoBuffer = ResourceDescriptorHeap[RayTraceCB.GeometryInfoBufferIdx];
    const GeometryInfo geoInfo = geoInfoBuffer[geometryIdx];

    StructuredBuffer<MeshVertex> vtxBuffer = ResourceDescriptorHeap[RayTraceCB.VtxBufferIdx];
    Buffer<uint> idxBuffer = ResourceDescriptorHeap[RayTraceCB.IdxBufferIdx];

    const uint primIdx = primitiveIdx;
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

void ClosestHitShaderStep0(
    in    MyPayload inPayload,
    inout ClosestHitShaderCheckpoint ckpt
) {
    ckpt.hasRay = false;
    const MeshVertex hitSurface = GetHitSurfaceMy(inPayload.attr, inPayload.gidx, inPayload.pidx);
    const Material material = GetGeometryMaterial(inPayload.gidx);
    
    if((!AppSettings.EnableDiffuse && !AppSettings.EnableSpecular) ||
        (!AppSettings.EnableDirect && !AppSettings.EnableIndirect))
    {
        ckpt.radiance = 0.0.xxx;
        ckpt.step = 7;
        return;
    }
    
    if (ckpt.pathLength > 1 && !AppSettings.EnableIndirect)
    {
        ckpt.radiance = 0.0.xxx;
        ckpt.step = 7;
        return;
    }
    
    float3x3 tangentToWorld = float3x3(hitSurface.Tangent, hitSurface.Bitangent, hitSurface.Normal);

    const float3 positionWS = hitSurface.Position;

    const float3 incomingRayOriginWS = inPayload.worldRayOrigin;
    const float3 incomingRayDirWS = inPayload.worldRayDirection;

    float3 normalWS = hitSurface.Normal;
    if(AppSettings.EnableNormalMaps)
    {
        // Sample the normal map, and convert the normal to world space
        Texture2D normalMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Normal)];

        float3 normalTS;
        normalTS.xy = normalMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xy * 2.0f - 1.0f;
        normalTS.z = sqrt(1.0f - saturate(normalTS.x * normalTS.x + normalTS.y * normalTS.y));
        normalWS = normalize(mul(normalTS, tangentToWorld));

        tangentToWorld._31_32_33 = normalWS;
    }
    ckpt.tangentToWorld = tangentToWorld;

    float3 baseColor = 1.0f;
    if(AppSettings.EnableAlbedoMaps && !AppSettings.EnableWhiteFurnaceMode)
    {
        Texture2D albedoMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Albedo)];
        baseColor = albedoMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;
    }

    Texture2D metallicMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Metallic)];
    const float metallic = saturate((AppSettings.EnableWhiteFurnaceMode ? 1.0f : metallicMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x) * AppSettings.MetallicScale);

    const bool enableDiffuse = (AppSettings.EnableDiffuse && metallic < 1.0f) || AppSettings.EnableWhiteFurnaceMode;
    const bool enableSpecular = (AppSettings.EnableSpecular && (AppSettings.EnableIndirectSpecular ? !(AppSettings.AvoidCausticPaths && ckpt.isDiffuse) : (ckpt.pathLength == 1)));
    
    ckpt.enableDiffuse = enableDiffuse;
    ckpt.enableSpecular = enableSpecular;

    if (enableDiffuse == false && enableSpecular == false)
    {
        ckpt.radiance = 0.0.xxx;
        ckpt.step = 7;
        return;
    }

    Texture2D roughnessMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Roughness)];
    const float sqrtRoughness = saturate((AppSettings.EnableWhiteFurnaceMode ? 1.0f : roughnessMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x) * AppSettings.RoughnessScale);

    const float3 diffuseAlbedo = lerp(baseColor, 0.0f, metallic) * (enableDiffuse ? 1.0f : 0.0f);
    const float3 specularAlbedo = lerp(0.03f, baseColor, metallic) * (enableSpecular ? 1.0f : 0.0f);
    float roughness = sqrtRoughness * sqrtRoughness;
    if(AppSettings.ClampRoughness)
        roughness = max(roughness, ckpt.roughness);

    float3 msEnergyCompensation = 1.0.xxx;
    if(AppSettings.ApplyMultiscatteringEnergyCompensation)
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
    ckpt.radiance = AppSettings.EnableWhiteFurnaceMode ? 0.0.xxx : emissiveMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).xyz;
    ckpt.step = 1;
    ckpt.incomingRayOriginWS = incomingRayOriginWS;
    ckpt.incomingRayDirWS = incomingRayDirWS;
    ckpt.normalWS = normalWS;
    ckpt.roughness = max(roughness, 0.001f);  // DIRTY FIX
    ckpt.sqrtRoughness = sqrt(ckpt.roughness);
    ckpt.positionWS = positionWS;
    ckpt.msEnergyCompensation = msEnergyCompensation;
    ckpt.diffuseAlbedo = diffuseAlbedo;
    ckpt.specularAlbedo = specularAlbedo;
}

void ClosestHitShaderStep1(
    inout ClosestHitShaderCheckpoint ckpt
)
{
    if (AppSettings.EnableSun && !AppSettings.EnableWhiteFurnaceMode)
    {
        float3 sunDirection = RayTraceCB.SunDirectionWS;

        if (AppSettings.SunAreaLightApproximation)
        {
            float3 D = RayTraceCB.SunDirectionWS;
            float3 R = reflect(ckpt.incomingRayDirWS, ckpt.normalWS);
            float r = RayTraceCB.SinSunAngularRadius;
            float d = RayTraceCB.CosSunAngularRadius;
            float DDotR = dot(D, R);
            float3 S = R - DDotR * D;
            sunDirection = DDotR < d ? normalize(d * D + normalize(S) * r) : R;
        }

        // Shoot a shadow ray to see if the sun is occluded
        RayDesc ray;
        ray.Origin = ckpt.positionWS;
        ray.Direction = RayTraceCB.SunDirectionWS;
        ray.TMin = 0.00001f;
        ray.TMax = FP32Max;

        ShadowPayload payload;
        payload.Visibility = 1.0f;

        uint traceRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

        // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
        if (ckpt.pathLength > AppSettings.MaxAnyHitPathLength)
            traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

        const uint hitGroupOffset = RayTypeShadow;
        const uint hitGroupGeoMultiplier = NumRayTypes;
        const uint missShaderIdx = RayTypeShadow;
        
        ckpt.hasRay = true;
        ckpt.hitGroupGeoMultiplier = hitGroupGeoMultiplier;
        ckpt.hitGroupOffset = hitGroupOffset;
        ckpt.missShaderIdx = missShaderIdx;
        ckpt.traceRayFlags = traceRayFlags;
        ckpt.nextRay = ray;
        ckpt.step = 2;
        ckpt.sunDirection = sunDirection;
    } else {
        ckpt.step = 3;
        ckpt.hasRay = false;
    }
}

void ClosestHitShaderStep2(
    in    MyPayload inPayload,
    inout ClosestHitShaderCheckpoint ckpt
)
{
    ckpt.radiance += CalcLighting_my(ckpt.normalWS, ckpt.sunDirection, RayTraceCB.SunIrradiance, ckpt.diffuseAlbedo, ckpt.specularAlbedo,
                                  ckpt.roughness, ckpt.positionWS, ckpt.incomingRayOriginWS, ckpt.msEnergyCompensation) * inPayload.visibility;
    ckpt.step = 3;
    ckpt.spotlightIndex = 0;
    ckpt.hasRay = false;
}

void ClosestHitShaderStep3(
    inout ClosestHitShaderCheckpoint ckpt
) {
    ckpt.hasRay = false;
    if (AppSettings.RenderLights) {
        if (ckpt.step == 3) {
            if (ckpt.spotlightIndex >= RayTraceCB.NumLights) {
                ckpt.step = 5;
                return;
            }

            SpotLight spotLight = LightCBuffer.Lights[ckpt.spotlightIndex];
            float3 surfaceToLight = spotLight.Position - ckpt.positionWS;
            float distanceToLight = length(surfaceToLight);
            surfaceToLight /= distanceToLight;
            float angleFactor = saturate(dot(surfaceToLight, spotLight.Direction));
            float angularAttenuation = smoothstep(spotLight.AngularAttenuationY, spotLight.AngularAttenuationX, angleFactor);

            float d = distanceToLight / spotLight.Range;
            float falloff = saturate(1.0f - (d * d * d * d));
            falloff = (falloff * falloff) / (distanceToLight * distanceToLight + 1.0f);

            angularAttenuation *= falloff;

            ckpt.step = 4;
            if (angularAttenuation > 0.0f)
            {
                // Shoot a shadow ray to see if the sun is occluded
                RayDesc ray;
                ray.Origin = ckpt.positionWS + ckpt.normalWS * 0.01f;
                ray.Direction = surfaceToLight;
                ray.TMin = SpotShadowNearClip;
                ray.TMax = distanceToLight - SpotShadowNearClip;

                ShadowPayload payload;
                payload.Visibility = 1.0f;

                uint traceRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;

                // Stop using the any-hit shader once we've hit the max path length, since it's *really* expensive
                if (ckpt.pathLength > AppSettings.MaxAnyHitPathLength)
                    traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

                ckpt.hitGroupOffset = RayTypeShadow;
                ckpt.hitGroupGeoMultiplier = NumRayTypes;
                ckpt.missShaderIdx = RayTypeShadow;
                ckpt.traceRayFlags = traceRayFlags;
                ckpt.hasRay = true;
                ckpt.step = 4;
                ckpt.nextRay = ray;
                ckpt.angularAttenuation = angularAttenuation;
                ckpt.sunDirection = surfaceToLight;
            } else {
                ckpt.hasRay = false;
            }
        }
    } else {
        ckpt.step = 5;
    }
}

void ClosestHitShaderStep4(
    in MyPayload inPayload,
    inout ClosestHitShaderCheckpoint ckpt
)
{
    if (ckpt.hasRay && inPayload.visibility > 0)
    {
        SpotLight spotLight = LightCBuffer.Lights[ckpt.spotlightIndex];
        float3 intensity = spotLight.Intensity * ckpt.angularAttenuation;
        ckpt.radiance += CalcLighting_my(ckpt.normalWS, ckpt.sunDirection, intensity, ckpt.diffuseAlbedo, ckpt.specularAlbedo,
                                      ckpt.roughness, ckpt.positionWS, ckpt.incomingRayOriginWS, ckpt.msEnergyCompensation) * inPayload.visibility;
    }
    ckpt.hasRay = false;
    ckpt.step = 3;
    ckpt.spotlightIndex ++;
}

void ClosestHitShaderStep5(
    inout ClosestHitShaderCheckpoint ckpt,
    out   ClosestHitShaderCheckpoint outCkpt
) {
    ckpt.hasRay = false;
    // Choose our next path by importance sampling our BRDFs
    float2 brdfSample = SamplePoint(ckpt.pixelIdx, ckpt.sampleSetIdx);

    float3 throughput = 0.0f;
    float3 rayDirTS = 0.0f;

    float selector = brdfSample.x;
    if(ckpt.enableSpecular == false)
        selector = 0.0f;
    else if(ckpt.enableDiffuse == false)
        selector = 1.0f;

    if(selector < 0.5f)
    {
        // We're sampling the diffuse BRDF, so sample a cosine-weighted hemisphere
        if(ckpt.enableSpecular)
            brdfSample.x *= 2.0f;
        rayDirTS = SampleDirectionCosineHemisphere(brdfSample.x, brdfSample.y);

        // The PDF of sampling a cosine hemisphere is NdotL / Pi, which cancels out those terms
        // from the diffuse BRDF and the irradiance integral
        throughput = ckpt.diffuseAlbedo;
    }
    else
    {
        // We're sampling the GGX specular BRDF by sampling the distribution of visible normals. See this post
        // for more info: https://schuttejoe.github.io/post/ggximportancesamplingpart2/.
        // Also see: https://hal.inria.fr/hal-00996995v1/document and https://hal.archives-ouvertes.fr/hal-01509746/document
        if(ckpt.enableDiffuse)
            brdfSample.x = (brdfSample.x - 0.5f) * 2.0f;

        float3 incomingRayDirTS = normalize(mul(ckpt.incomingRayDirWS, transpose(ckpt.tangentToWorld)));
        float3 microfacetNormalTS = SampleGGXVisibleNormal(-incomingRayDirTS, ckpt.roughness, ckpt.roughness, brdfSample.x, brdfSample.y);
        float3 sampleDirTS = reflect(incomingRayDirTS, microfacetNormalTS);

        float3 normalTS = float3(0.0f, 0.0f, 1.0f);

        float3 F = AppSettings.EnableWhiteFurnaceMode ? 1.0.xxx : Fresnel(ckpt.specularAlbedo, microfacetNormalTS, sampleDirTS);
        float G1 = SmithGGXMasking(normalTS, sampleDirTS, -incomingRayDirTS, ckpt.roughness * ckpt.roughness);
        float G2 = SmithGGXMaskingShadowing(normalTS, sampleDirTS, -incomingRayDirTS, ckpt.roughness * ckpt.roughness);

        throughput = (F * (G2 / G1));
        rayDirTS = sampleDirTS;

        if(AppSettings.ApplyMultiscatteringEnergyCompensation)
        {
            float2 DFG = GGXEnvironmentBRDFScaleBias(saturate(dot(normalTS, -ckpt.incomingRayDirWS)), ckpt.sqrtRoughness);

            // Improve energy preservation by applying a scaled version of the original
            // single scattering specular lobe. Based on "Practical multiple scattering
            // compensation for microfacet models" [Turquin19].
            //
            // See: https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
            float Ess = DFG.x;
            throughput *= 1.0.xxx + ckpt.specularAlbedo * (1.0f / Ess - 1.0f);
        }
    }

    const float3 rayDirWS = normalize(mul(rayDirTS, ckpt.tangentToWorld));

    if(ckpt.enableDiffuse && ckpt.enableSpecular)
        throughput *= 2.0f;

    // Shoot another ray to get the next path
    RayDesc ray;
    ray.Origin = ckpt.positionWS;
    ray.Direction = rayDirWS;
    ray.TMin = 0.00001f;
    ray.TMax = FP32Max;
    
    if (ckpt.pathLength == 1 && !AppSettings.EnableDirect)
    {
        ckpt.radiance = 0.0.xxx;
        ckpt.step = 7;
        return;
    }
    
    ckpt.rayDirWS = rayDirWS;
    ckpt.throughput = throughput;
    
    if(AppSettings.EnableIndirect && 
       (ckpt.pathLength + 1 < AppSettings.MaxPathLength) &&
       !AppSettings.EnableWhiteFurnaceMode)
    {
        ckpt.hasNewCkpt = true;
        
        outCkpt = (ClosestHitShaderCheckpoint)0;
        outCkpt.radiance = 0;
        outCkpt.pathLength = ckpt.pathLength + 1;
        outCkpt.pixelIdx = ckpt.pixelIdx;
        outCkpt.sampleSetIdx = ckpt.sampleSetIdx;
        outCkpt.isDiffuse = (selector < 0.5f);
        outCkpt.roughness = ckpt.roughness;
        
        outCkpt.traceRayFlags = 0;
        if (outCkpt.pathLength > AppSettings.MaxAnyHitPathLength)
        {
            outCkpt.traceRayFlags = RAY_FLAG_FORCE_OPAQUE;
        }
        outCkpt.hitGroupOffset = RayTypeRadiance;
        outCkpt.hitGroupGeoMultiplier = NumRayTypes;
        outCkpt.missShaderIdx = RayTypeRadiance;
        outCkpt.hasRay = true;
        outCkpt.nextRay = ray;
        ckpt.step = 7;  // Branches join
        ckpt.isChildRayRadiance = true;
    }
    else
    {
        // SHADOW RAY
        ckpt.hasNewCkpt = false;
        ckpt.hasRay = true;
        ckpt.traceRayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
        if (ckpt.pathLength + 1 > AppSettings.MaxAnyHitPathLength)
            ckpt.traceRayFlags = RAY_FLAG_FORCE_OPAQUE;
        
        ckpt.hitGroupOffset = RayTypeShadow;
        ckpt.hitGroupGeoMultiplier = NumRayTypes;
        ckpt.missShaderIdx = RayTypeShadow;
        ckpt.nextRay = ray;
        ckpt.step = 6;  // Just continue
        ckpt.isChildRayRadiance = false;
    }
}

[shader("raygeneration")]
void RaygenShader()
{
    const uint2 pixelCoord = DispatchRaysIndex().xy;
    const uint pixelIdx = pixelCoord.y * DispatchRaysDimensions().x + pixelCoord.x;

    uint sampleSetIdx = 0;

    // Form a primary ray by un-projecting the pixel coordinate using the inverse view * projection matrix
    float2 primaryRaySample = SamplePoint(pixelIdx, sampleSetIdx);

    float2 rayPixelPos = pixelCoord + primaryRaySample;
    float2 ncdXY = (rayPixelPos / (DispatchRaysDimensions().xy * 0.5f)) - 1.0f;
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
    
    ClosestHitShaderCheckpoint ckpts[4];
    ckpts[0].radiance = float3(0,0,0);
    int ckptIdx = 0;
    
    ClosestHitShaderCheckpoint ckpt = (ClosestHitShaderCheckpoint)0;
    ckpt.nextRay = ray;
    ckpt.hitGroupOffset = RayTypeRadiance;
    ckpt.hitGroupGeoMultiplier = NumRayTypes;
    ckpt.missShaderIdx = RayTypeRadiance;
    ckpt.hasRay = true;
    ckpt.hasNewCkpt = false;
    ckpt.step = 0;
    ckpt.pathLength = 1;
    ckpt.pixelIdx = pixelIdx;
    ckpt.sampleSetIdx = sampleSetIdx;
    
    ckpts[0] = ckpt;
    ckptIdx++;
    float3 ret = float3(0,0,0);
    
    while (ckptIdx> 0) {
        ckpt = ckpts[--ckptIdx];
        ckpt.hasNewCkpt = false;
        
        MyPayload mypayload;

        if (ckpt.hasRay)
        {
            mypayload.gidx = 0xFFFFFFFF;
            mypayload.pidx = 0xFFFFFFFF;
            mypayload.visibility = 1.0f;
            mypayload.PathLength = ckpt.pathLength;
            TraceRay(Scene, ckpt.traceRayFlags, 0xFFFFFFFF, ckpt.hitGroupOffset, ckpt.hitGroupGeoMultiplier, ckpt.missShaderIdx, ckpt.nextRay, mypayload);
        }
        
        switch (ckpt.step) {
            case 0: {
                if (mypayload.pidx != 0xFFFFFFFF) {
                    ClosestHitShaderStep0(mypayload, ckpt);
                    ckpts[ckptIdx++] = ckpt;
                }
                else
                {
                    ckpt.radiance = mypayload.radiance;
                    ckpt.step = 7;
                    ckpt.hasRay = false;
                    ckpts[ckptIdx++] = ckpt;
                }
                break;
            }
            case 1: {
                ClosestHitShaderStep1(ckpt);
                ckpts[ckptIdx++] = ckpt;
                break;
            }
            case 2: {
                ClosestHitShaderStep2(mypayload, ckpt);
                ckpts[ckptIdx++] = ckpt;
                break;
            }
            case 3: {
                ClosestHitShaderStep3(ckpt);
                ckpts[ckptIdx++] = ckpt;
                break;
            }
            case 4: {
                ClosestHitShaderStep4(mypayload, ckpt);
                ckpts[ckptIdx++] = ckpt;
                break;
            }
            case 5:{
                ClosestHitShaderCheckpoint outCkpt;
                ClosestHitShaderStep5(ckpt, outCkpt);
                ckpts[ckptIdx++] = ckpt;
                if (ckpt.hasNewCkpt)
                {
                    ckpts[ckptIdx++] = outCkpt;
                }
                break;
            }
            case 6: {
                if (AppSettings.EnableWhiteFurnaceMode)
                {
                    ckpt.radiance = ckpt.throughput;
                }
                else
                {
                    TextureCube skyTexture = TexCubeTable[RayTraceCB.SkyTextureIdx];
                    float3 skyRadiance = AppSettings.EnableSky ? skyTexture.SampleLevel(LinearSampler, ckpt.rayDirWS, 0.0f).xyz : 0.0.xxx;
                    ckpt.radiance += mypayload.visibility * skyRadiance * ckpt.throughput;
                }
                ckpt.step = 7;
                ckpt.hasRay = false;
                ckpt.hasNewCkpt = false;
                ckpts[ckptIdx++] = ckpt;
                break;
            }
            case 7: {
                if (ckptIdx > 0) {
                    ClosestHitShaderCheckpoint ckptPrev = ckpts[ckptIdx - 1];
                    ckptPrev.radiance += ckpt.radiance * ckptPrev.throughput;
                    ckpts[ckptIdx - 1] = ckptPrev;
                }
                break;
            }
        }
    }

    /*
    PrimaryPayload payload;
    payload.Radiance = 0.0f;
    payload.Roughness = 0.0f;
    payload.PathLength = 1;       // 6b
    payload.PixelIdx = pixelIdx;  // 24b
    payload.SampleSetIdx = sampleSetIdx;
    payload.IsDiffuse = false;    // 1b

    const uint hitGroupOffset = RayTypeRadiance;
    const uint hitGroupGeoMultiplier = NumRayTypes;
    const uint missShaderIdx = RayTypeRadiance;
    TraceRay(Scene, traceRayFlags, 0xFFFFFFFF, hitGroupOffset, hitGroupGeoMultiplier, missShaderIdx, ray, payload);

    payload.Radiance = clamp(payload.Radiance, 0.0f, FP16Max);
    */
    ret = ckpts[0].radiance;
    if (any(isnan(ret)) || any(isinf(ret)))
        ret = 0.0.xxx;
    ret = clamp(ret, 0.0f, FP16Max);

    // Update the progressive result with the new radiance sample
    const float lerpFactor = RayTraceCB.CurrSampleIdx / (RayTraceCB.CurrSampleIdx + 1.0f);
    float3 newSample = ret;
    float3 currValue = RenderTarget[pixelCoord].xyz;
    float3 newValue = lerp(newSample, currValue, lerpFactor);
    
    if (RayTraceCB.CurrSampleIdx == 0) {  // Prevent NAN from sticking
        newValue = newSample;
    }
    
    RenderTarget[pixelCoord] = float4(newValue, 1.0f);
}

// My new closest hit shader = just log stuff
[shader("closesthit")]
void ClosestHitShader(inout MyPayload payload, in HitAttributes attr)
{
    payload.gidx = GeometryIndex();
    payload.pidx = PrimitiveIndex();
    payload.attr = attr;
    payload.worldRayDirection = WorldRayDirection();
    payload.worldRayOrigin = WorldRayOrigin();
}

[shader("anyhit")]
void AnyHitShader(inout MyPayload payload, in HitAttributes attr)
{
    const MeshVertex hitSurface = GetHitSurface(attr, GeometryIndex());
    const Material material = GetGeometryMaterial(GeometryIndex());

    // Standard alpha testing
    Texture2D opacityMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Opacity)];
    if(opacityMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x < 0.35f)
        IgnoreHit();
}

[shader("anyhit")]
void ShadowAnyHitShader(inout MyPayload payload, in HitAttributes attr)
{
    const MeshVertex hitSurface = GetHitSurface(attr, GeometryIndex());
    const Material material = GetGeometryMaterial(GeometryIndex());

    // Standard alpha testing
    Texture2D opacityMap = ResourceDescriptorHeap[NonUniformResourceIndex(material.Opacity)];
    if(opacityMap.SampleLevel(MeshSampler, hitSurface.UV, 0.0f).x < 0.35f)
        IgnoreHit();
}

[shader("miss")]
void MissShader(inout MyPayload payload)
{
    if(AppSettings.EnableWhiteFurnaceMode)
    {
        payload.radiance = 1.0.xxx;
    }
    else
    {
        const float3 rayDir = WorldRayDirection();

        TextureCube skyTexture = ResourceDescriptorHeap[RayTraceCB.SkyTextureIdx];
        payload.radiance = AppSettings.EnableSky ? skyTexture.SampleLevel(LinearSampler, rayDir, 0.0f).xyz : 0.0.xxx;

        if(payload.PathLength == 1)
        {
            float cosSunAngle = dot(rayDir, RayTraceCB.SunDirectionWS);
            if(cosSunAngle >= RayTraceCB.CosSunAngularRadius)
                payload.radiance = RayTraceCB.SunRenderColor;
        }
    }
}

[shader("closesthit")]
void ShadowHitShader(inout MyPayload payload, in HitAttributes attr)
{
    payload.visibility = 0.0f;
}

[shader("miss")]
void ShadowMissShader(inout MyPayload payload)
{
    payload.visibility = 1.0f;
}

[shader("closesthit")]
void ShadowHitShader_old(inout ShadowPayload payload, in HitAttributes attr)
{
    payload.Visibility = 0.0f;
}

[shader("miss")]
void ShadowMissShader_old(inout ShadowPayload payload)
{
    payload.Visibility = 1.0f;
}
