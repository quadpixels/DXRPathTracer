struct RayGenWorkInfo
{
    uint DispatchWidth;
    uint DispatchHeight;
    uint DispatchIndex;
    uint DispatchStride;
    uint TileCountX;
    uint TileCountY;
    uint TileArea;
    uint StaticWorkItemLimit;
    uint AtomicWorkItemCount;
    uint StaticWorkItemStep;
    bool Tiled;
    bool Atomic;
};

static RayGenWorkInfo MakeRayGenWorkInfo(uint3 dispatchIndex)
{
    RayGenWorkInfo info;
    const uint dispatchWidth = DispatchRaysDimensions().x;
    const uint dispatchHeight = DispatchRaysDimensions().y;

    info.DispatchWidth = dispatchWidth;
    info.DispatchHeight = dispatchHeight;
    info.DispatchIndex = dispatchIndex.y * dispatchWidth + dispatchIndex.x;
    info.DispatchStride = dispatchWidth * dispatchHeight;
    info.Tiled = (RayTraceCB.myFlags & 16u) != 0u;
    info.Atomic = (RayTraceCB.myFlags & 64u) != 0u;
    info.TileCountX = (RayTraceCB.DispatchWidth + dispatchWidth - 1u) / dispatchWidth;
    info.TileCountY = (RayTraceCB.DispatchHeight + dispatchHeight - 1u) / dispatchHeight;
    info.TileArea = dispatchWidth * dispatchHeight;
    info.StaticWorkItemLimit = info.Tiled ? info.TileCountX * info.TileCountY : RayTraceCB.TotalNumPixels;
    info.AtomicWorkItemCount = info.Tiled ? info.TileCountX * info.TileCountY * info.TileArea : RayTraceCB.TotalNumPixels;
    info.StaticWorkItemStep = info.Tiled ? 1u : info.DispatchStride;
    return info;
}

static bool AcquireRayGenWork(in RayGenWorkInfo info, inout uint workItem)
{
    if(info.Atomic)
    {
        uint waveBaseIdx = 0;
        const uint laneIdx = WavePrefixCountBits(true);
        const uint waveSize = WaveActiveCountBits(true);
        if(WaveIsFirstLane())
            InterlockedAdd(WavefrontCounters[Counter_WorkCursor], waveSize, waveBaseIdx);
        waveBaseIdx = WaveReadLaneFirst(waveBaseIdx);
        if(waveBaseIdx >= info.AtomicWorkItemCount)
            return false;
        workItem = waveBaseIdx + laneIdx;
        return true;
    }

    return workItem < info.StaticWorkItemLimit;
}

static bool ResolveRayGenPixel(in RayGenWorkInfo info, uint workItem, out uint pixelIdx, out uint2 pixelCoord)
{
    pixelIdx = 0;
    pixelCoord = 0;
    if(info.Tiled)
    {
        if(info.Atomic)
        {
            const uint tileLinearIdx = workItem / info.TileArea;
            const uint localIdx = workItem % info.TileArea;
            const uint tileX = tileLinearIdx % info.TileCountX;
            const uint tileY = tileLinearIdx / info.TileCountX;
            pixelCoord = uint2(localIdx % info.DispatchWidth, localIdx / info.DispatchWidth) +
                         uint2(tileX * info.DispatchWidth, tileY * info.DispatchHeight);
        }
        else
        {
            const uint tileX = workItem % info.TileCountX;
            const uint tileY = workItem / info.TileCountX;
            const uint2 dispatchIndex = uint2(info.DispatchIndex % info.DispatchWidth,
                                              info.DispatchIndex / info.DispatchWidth);
            pixelCoord = dispatchIndex + uint2(tileX * info.DispatchWidth, tileY * info.DispatchHeight);
        }

        if(pixelCoord.x >= RayTraceCB.DispatchWidth || pixelCoord.y >= RayTraceCB.DispatchHeight)
            return false;

        pixelIdx = pixelCoord.y * RayTraceCB.DispatchWidth + pixelCoord.x;
        return true;
    }

    pixelIdx = workItem;
    pixelCoord = uint2(pixelIdx % RayTraceCB.DispatchWidth, pixelIdx / RayTraceCB.DispatchWidth);
    return true;
}

static void TraceRayGenPixel(uint pixelIdx, uint2 pixelCoord)
{
    uint sampleSetIdx = 0;

    float2 primaryRaySample = SamplePoint(pixelIdx, sampleSetIdx);
    float2 rayPixelPos = pixelCoord + primaryRaySample;
    float2 ncdXY = (rayPixelPos / (float2(RayTraceCB.DispatchWidth, RayTraceCB.DispatchHeight) * 0.5f)) - 1.0f;
    ncdXY.y *= -1.0f;
    float4 rayStart = mul(float4(ncdXY, 0.0f, 1.0f), RayTraceCB.InvViewProjection);
    float4 rayEnd = mul(float4(ncdXY, 1.0f, 1.0f), RayTraceCB.InvViewProjection);

    rayStart.xyz /= rayStart.w;
    rayEnd.xyz /= rayEnd.w;

    RayDesc ray;
    ray.Origin = rayStart.xyz;
    ray.Direction = normalize(rayEnd.xyz - rayStart.xyz);
    ray.TMin = 0.0f;
    ray.TMax = length(rayEnd.xyz - rayStart.xyz);

    PrimaryPayload payload;
    payload.Radiance = 0.0f;
    payload.Roughness = 0.0f;
    payload.PathLength = 1;
    payload.PixelIdx = pixelIdx;
    payload.SampleSetIdx = sampleSetIdx;
    payload.IsDiffuse = false;

    uint traceRayFlags = 0;
    if(payload.PathLength > AppSettings.MaxAnyHitPathLength)
        traceRayFlags = RAY_FLAG_FORCE_OPAQUE;

    TraceRay(Scene, traceRayFlags, 0xFFFFFFFF, RayTypeRadiance, NumRayTypes,
             RayTypeRadiance, ray, payload);

    payload.Radiance = clamp(payload.Radiance, 0.0f, FP16Max);
    const float lerpFactor = RayTraceCB.CurrSampleIdx / (RayTraceCB.CurrSampleIdx + 1.0f);
    float3 newSample = payload.Radiance;
    float3 currValue = RenderTarget[pixelCoord].xyz;
    float3 newValue = lerp(newSample, currValue, lerpFactor);
    if(RayTraceCB.CurrSampleIdx == 0)
        newValue = newSample;

    RenderTarget[pixelCoord] = float4(newValue, 1.0f);
}
