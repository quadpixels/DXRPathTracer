    case 11:
    case 13:
    case 15:
    case 16: {
      const bool tiledPersistentWarp = activeRenderPath == 13 || activeRenderPath == 16;
      const bool atomicPersistentWarp = activeRenderPath == 15 || activeRenderPath == 16;
      const char* profileName = atomicPersistentWarp ?
          (tiledPersistentWarp ? "TraceRay DispatchRays (DXR 1.0 Tiled Persistent Warp, Atomic)" : "TraceRay DispatchRays (DXR 1.0 Persistent Warp, Atomic)") :
          (tiledPersistentWarp ? "TraceRay DispatchRays (DXR 1.0 Tiled Persistent Warp)" : "TraceRay DispatchRays (DXR 1.0 Persistent Warp)");
      ProfileBlock pb(cmdList, profileName);
      cmdList->SetPipelineState1(rtPSO);
      D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
      dispatchDesc.HitGroupTable = rtHitTable.ShaderTable();
      dispatchDesc.MissShaderTable = rtMissTable.ShaderTable();
      dispatchDesc.RayGenerationShaderRecord = rtRayGenTable.ShaderRecord(0);
      dispatchDesc.Width = ActiveWavefrontThreadGroupSize();
      dispatchDesc.Height = uint32(Max(g_persistent_worker_groups_actual, 1));
      dispatchDesc.Depth = 1;
      if(atomicPersistentWarp)
      {
          D3D12_CPU_DESCRIPTOR_HANDLE atomicPersistentUAVs[] =
          {
              rtTarget.UAV,
              DX12::NullStructuredBufferUAV,
              DX12::NullStructuredBufferUAV,
              DX12::NullStructuredBufferUAV,
              DX12::NullStructuredBufferUAV,
              wavefrontCounterBuffer.UAV,
              DX12::NullStructuredBufferUAV,
              DX12::NullStructuredBufferUAV,
              DX12::NullRawBufferUAV,
          };
          DX12::BindTempDescriptorTable(cmdList, atomicPersistentUAVs, ArraySize_(atomicPersistentUAVs),
                                        RTParams_UAVDescriptor, CmdListMode::Compute);

          constexpr uint32 maxRayGenIterations = 4;
          const uint64 workerCount = uint64(dispatchDesc.Width) * uint64(dispatchDesc.Height);
          const uint64 totalWorkItems = uint64(rtTarget.Width()) * uint64(rtTarget.Height());
          const uint32 dispatchCount = uint32((totalWorkItems + workerCount * maxRayGenIterations - 1) /
                                               (workerCount * maxRayGenIterations)) + 1;

          rtConstants.PersistentRayGenMaxIterations = maxRayGenIterations;
          DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);
          wavefrontCounterBuffer.Transition(cmdList, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
          clearWavefrontCounters();
          cmdList->SetPipelineState1(rtPSO);

          for(uint32 dispatchIdx = 0; dispatchIdx < dispatchCount; ++dispatchIdx)
          {
              DX12::CmdList->DispatchRays(&dispatchDesc);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
              rtTarget.UAVBarrier(cmdList);
          }
      }
      else
      {
          rtConstants.PersistentRayGenMaxIterations = 0;
          DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);
          DX12::CmdList->DispatchRays(&dispatchDesc);
      }
      if(atomicPersistentWarp)
      {
          wavefrontCounterBuffer.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
      }
      break;
    }
