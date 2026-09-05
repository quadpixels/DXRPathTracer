//=================================================================================================
//
//  DXR Path Tracer
//  by MJP
//  http://mynameismjp.wordpress.com/
//
//  All code and content licensed under the MIT license
//
//=================================================================================================

#pragma once

#include <PCH.h>

#include <App.h>
#include <InterfacePointers.h>
#include <Input.h>
#include <Graphics/Camera.h>
#include <Graphics/Model.h>
#include <Graphics/Skybox.h>
#include <Graphics/GraphicsTypes.h>

#include "PostProcessor.h"
#include "MeshRenderer.h"

using namespace SampleFramework12;

class DXRPathTracer : public App
{

protected:

    FirstPersonCamera camera;

    Skybox skybox;
    SkyCache skyCache;

    PostProcessor postProcessor;

    // Model
    Model sceneModels[uint64(Scenes::NumValues)];
    const Model* currentModel = nullptr;
    MeshRenderer meshRenderer;

    RenderTexture mainTarget;
    RenderTexture resolveTarget;
    RenderTexture deferredMSAATarget;
    DepthBuffer depthBuffer;

    Array<SpotLight> spotLights;
    ConstantBuffer spotLightBuffer;
    StructuredBuffer spotLightBoundsBuffer;
    StructuredBuffer spotLightInstanceBuffer;
    RawBuffer spotLightClusterBuffer;
    uint64 numIntersectingSpotLights = 0;

    ID3D12RootSignature* clusterRS = nullptr;
    CompiledShaderPtr clusterVS;
    CompiledShaderPtr clusterFrontFacePS;
    CompiledShaderPtr clusterBackFacePS;
    CompiledShaderPtr clusterIntersectingPS;
    ID3D12PipelineState* clusterFrontFacePSO = nullptr;
    ID3D12PipelineState* clusterBackFacePSO = nullptr;
    ID3D12PipelineState* clusterIntersectingPSO = nullptr;
    RenderTexture clusterMSAATarget;

    StructuredBuffer spotLightClusterVtxBuffer;
    FormattedBuffer spotLightClusterIdxBuffer;
    Array<Float3> coneVertices;

    CompiledShaderPtr fullScreenTriVS;
    CompiledShaderPtr resolvePS[NumMSAAModes];
    ID3D12RootSignature* resolveRootSignature = nullptr;
    ID3D12PipelineState* resolvePSO = nullptr;

    bool32 stablePowerState = false;

    // Ray tracing resources
    CompiledShaderPtr rayTraceLib;
    CompiledShaderPtr rayTraceLib_SER;
    CompiledShaderPtr rayTraceLibLoop_SER;
    CompiledShaderPtr rayTraceLibLoop_my;
    RenderTexture rtTarget;
    ID3D12RootSignature* rtRootSignature = nullptr;
    ID3D12StateObject* rtPSO = nullptr;
    ID3D12StateObject* rtPSO_SER = nullptr;
    ID3D12StateObject* rtPSOLoop_SER = nullptr;
    ID3D12StateObject* rtPSOLoop_my = nullptr;
    bool buildAccelStructure = true;
    uint64 lastBuildAccelStructureFrame = uint64(-1);
    RawBuffer rtBottomLevelAccelStructure;
    RawBuffer rtTopLevelAccelStructure;
    StructuredBuffer rtRayGenTable, rtRayGenTable_SER, rtRayGenTableLoop_SER, rtRayGenTableLoop_my;
    StructuredBuffer rtHitTable, rtHitTable_SER, rtHitTableLoop_SER, rtHitTableLoop_my;
    StructuredBuffer rtMissTable, rtMissTable_SER, rtMissTableLoop_SER, rtMissTableLoop_my;
    StructuredBuffer rtGeoInfoBuffer;
    FirstPersonCamera rtCurrCamera;
    bool rtShouldRestartPathTrace = false;
    uint32 rtCurrSampleIdx = 0;

    // Something else
    CompiledShaderPtr rayTraceRayQueryCS;
    ID3D12PipelineState* rtRayQueryPSO{};
    CompiledShaderPtr rayTraceRayQuery1CS;
    ID3D12PipelineState* rtRayQuery1PSO{};
    CompiledShaderPtr wavefrontClearCS;
    CompiledShaderPtr wavefrontGeneratePrimaryCS;
    CompiledShaderPtr wavefrontTraceHitsCS;
    CompiledShaderPtr wavefrontShadeHitsCS;
    CompiledShaderPtr wavefrontTraceShadowsCS;
    CompiledShaderPtr wavefrontPrepareDispatchArgsCS;
    CompiledShaderPtr wavefrontClearReorderCS;
    CompiledShaderPtr wavefrontCountReorderBinsCS;
    CompiledShaderPtr wavefrontPrefixReorderBinsCS;
    CompiledShaderPtr wavefrontScatterReorderedRaysCS;
    CompiledShaderPtr wavefrontAdvanceCS;
    CompiledShaderPtr wavefrontAccumulateCS;
    ID3D12PipelineState* wavefrontClearPSO = nullptr;
    ID3D12PipelineState* wavefrontGeneratePrimaryPSO = nullptr;
    ID3D12PipelineState* wavefrontTraceHitsPSO = nullptr;
    ID3D12PipelineState* wavefrontShadeHitsPSO = nullptr;
    ID3D12PipelineState* wavefrontTraceShadowsPSO = nullptr;
    ID3D12PipelineState* wavefrontPrepareDispatchArgsPSO = nullptr;
    ID3D12PipelineState* wavefrontClearReorderPSO = nullptr;
    ID3D12PipelineState* wavefrontCountReorderBinsPSO = nullptr;
    ID3D12PipelineState* wavefrontPrefixReorderBinsPSO = nullptr;
    ID3D12PipelineState* wavefrontScatterReorderedRaysPSO = nullptr;
    ID3D12PipelineState* wavefrontAdvancePSO = nullptr;
    ID3D12PipelineState* wavefrontAccumulatePSO = nullptr;
    StructuredBuffer wavefrontPathStateBuffer;
    StructuredBuffer wavefrontRayQueueA;
    StructuredBuffer wavefrontRayQueueB;
    StructuredBuffer wavefrontShadowQueue;
    StructuredBuffer wavefrontCounterBuffer;
    StructuredBuffer wavefrontHitQueueA;
    StructuredBuffer wavefrontHitQueueB;
    RawBuffer wavefrontDispatchArgsBuffer;
    ID3D12CommandSignature* wavefrontDispatchCommandSignature = nullptr;

    virtual void Initialize() override;
    virtual void Shutdown() override;

    virtual void Render(const Timer& timer) override;
    virtual void Update(const Timer& timer) override;

    virtual void BeforeReset() override;
    virtual void AfterReset() override;

    virtual void CreatePSOs() override;
    virtual void DestroyPSOs() override;

    void CreateRenderTargets();
    void InitializeScene();

    void InitRayTracing();
    void CreateRayTracingPSOs(const CompiledShaderPtr& shader_ptr, ID3D12StateObject** rtpso,
      StructuredBuffer* raygen_table, StructuredBuffer* hit_table, StructuredBuffer* miss_table);
    void CreateRayTracingRayQueryPSOs();

    void UpdateLights();

    void RenderClusters();
    void RenderForward();
    void RenderResolve();
    void RenderRayTracing();
    void RenderHUD(const Timer& timer);

    void BuildRTAccelerationStructure();

public:

    DXRPathTracer(const wchar* cmdLine);
};
