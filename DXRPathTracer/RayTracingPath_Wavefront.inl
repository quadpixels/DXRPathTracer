    case 6:
    case 9: {
      const bool gpuWavefrontPath = activeRenderPath == 9;
      ProfileBlock pb(cmdList, gpuWavefrontPath ? "RayQuery GPU Wavefront Dispatch" : "RayQuery Wavefront Dispatch");

      static const char* TraceHitProfileNames[] =
      {
          "WF Trace Hits B0",
          "WF Trace Hits B1",
          "WF Trace Hits B2",
          "WF Trace Hits B3",
          "WF Trace Hits B4",
          "WF Trace Hits B5",
          "WF Trace Hits B6",
          "WF Trace Hits B7",
      };

      static const char* HitSortProfileNames[] =
      {
          "WF Hit Sort B0",
          "WF Hit Sort B1",
          "WF Hit Sort B2",
          "WF Hit Sort B3",
          "WF Hit Sort B4",
          "WF Hit Sort B5",
          "WF Hit Sort B6",
          "WF Hit Sort B7",
      };

      static const char* PrepareAfterTraceProfileNames[] =
      {
          "WF Prepare Args After Trace B0",
          "WF Prepare Args After Trace B1",
          "WF Prepare Args After Trace B2",
          "WF Prepare Args After Trace B3",
          "WF Prepare Args After Trace B4",
          "WF Prepare Args After Trace B5",
          "WF Prepare Args After Trace B6",
          "WF Prepare Args After Trace B7",
      };

      static const char* PrepareAfterShadeProfileNames[] =
      {
          "WF Prepare Args After Shade B0",
          "WF Prepare Args After Shade B1",
          "WF Prepare Args After Shade B2",
          "WF Prepare Args After Shade B3",
          "WF Prepare Args After Shade B4",
          "WF Prepare Args After Shade B5",
          "WF Prepare Args After Shade B6",
          "WF Prepare Args After Shade B7",
      };

      static const char* ShadeHitProfileNames[] =
      {
          "WF Shade Hits B0",
          "WF Shade Hits B1",
          "WF Shade Hits B2",
          "WF Shade Hits B3",
          "WF Shade Hits B4",
          "WF Shade Hits B5",
          "WF Shade Hits B6",
          "WF Shade Hits B7",
      };

      static const char* TraceShadowProfileNames[] =
      {
          "WF Trace Shadows B0",
          "WF Trace Shadows B1",
          "WF Trace Shadows B2",
          "WF Trace Shadows B3",
          "WF Trace Shadows B4",
          "WF Trace Shadows B5",
          "WF Trace Shadows B6",
          "WF Trace Shadows B7",
      };

      static const char* AdvanceProfileNames[] =
      {
          "WF Advance B0",
          "WF Advance B1",
          "WF Advance B2",
          "WF Advance B3",
          "WF Advance B4",
          "WF Advance B5",
          "WF Advance B6",
          "WF Advance B7",
      };

      D3D12_CPU_DESCRIPTOR_HANDLE uavs[] =
      {
          rtTarget.UAV,
          wavefrontPathStateBuffer.UAV,
          wavefrontRayQueueA.UAV,
          wavefrontRayQueueB.UAV,
          wavefrontShadowQueue.UAV,
          wavefrontCounterBuffer.UAV,
          wavefrontHitQueueA.UAV,
          wavefrontHitQueueB.UAV,
          wavefrontDispatchArgsBuffer.UAV,
      };
      DX12::BindTempDescriptorTable(cmdList, uavs, ArraySize_(uavs), RTParams_UAVDescriptor, CmdListMode::Compute);

      const uint32 width = uint32(rtTarget.Width());
      const uint32 height = uint32(rtTarget.Height());
      const uint32 gx = (width + 7) / 8;
      const uint32 gy = (height + 7) / 8;
      const uint64 dispatchArgsStride = sizeof(D3D12_DISPATCH_ARGUMENTS);
      const uint64 currentRayDispatchArgsOffset = 0 * dispatchArgsStride;
      const uint64 hitDispatchArgsOffset = 1 * dispatchArgsStride;
      const uint64 shadowDispatchArgsOffset = 2 * dispatchArgsStride;
      const uint64 hitMetaDispatchArgsOffset = 3 * dispatchArgsStride;
      const uint64 sortHitDispatchArgsOffset = 4 * dispatchArgsStride;
      const uint64 scatterHitDispatchArgsOffset = 5 * dispatchArgsStride;
      D3D12_RESOURCE_STATES dispatchArgsState = D3D12_RESOURCE_STATE_COMMON;

      auto bindWavefrontConstants = [&](uint32 bounce, uint32 readQueue, uint32 writeQueue)
      {
          const bool sortThisBounce = g_wavefront_reorder && (bounce > 0 || g_wavefront_skip_primary_sort == false);

          rtConstants.myFlags &= ~(2u | 4u);
          if(sortThisBounce && g_wavefront_block_sort == false)
              rtConstants.myFlags |= 2u;
          else if(sortThisBounce && g_wavefront_block_sort)
              rtConstants.myFlags |= 4u;

          rtConstants.WavefrontBounce = bounce;
          rtConstants.WavefrontReadQueue = readQueue;
          rtConstants.WavefrontWriteQueue = writeQueue;
          DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);
      };

      auto prepareWavefrontDispatchArgs = [&](const char* profileName)
      {
          ProfileBlock preparePB(cmdList, profileName);

          if(dispatchArgsState != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
          {
              wavefrontDispatchArgsBuffer.Transition(cmdList, dispatchArgsState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
              dispatchArgsState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
          }

          cmdList->SetPipelineState(wavefrontPrepareDispatchArgsPSO);
          DX12::CmdList->Dispatch(1, 1, 1);
          wavefrontDispatchArgsBuffer.UAVBarrier(cmdList);
          wavefrontDispatchArgsBuffer.Transition(cmdList, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
          dispatchArgsState = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
      };

      bindWavefrontConstants(0, 0, 1);
      clearWavefrontCounters();

      {
          ProfileBlock generatePB(cmdList, "WF Generate Primary");
          cmdList->SetPipelineState(wavefrontGeneratePrimaryPSO);
          DX12::CmdList->Dispatch(gx, gy, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          wavefrontRayQueueA.UAVBarrier(cmdList);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
      }
      prepareWavefrontDispatchArgs("WF Prepare Args Initial");

      uint32 currentQueue = 0;
      for(uint32 bounce = 0; bounce < uint32(AppSettings::MaxPathLength); ++bounce)
      {
          const uint32 nextQueue = currentQueue ^ 1;
          bindWavefrontConstants(bounce, currentQueue, nextQueue);

          {
              ProfileBlock tracePB(cmdList, TraceHitProfileNames[bounce]);
              cmdList->SetPipelineState(wavefrontTraceHitsPSOVariants[wavefrontVariantIdx]);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       currentRayDispatchArgsOffset, nullptr, 0);
              wavefrontPathStateBuffer.UAVBarrier(cmdList);
              wavefrontRayQueueA.UAVBarrier(cmdList);
              wavefrontRayQueueB.UAVBarrier(cmdList);
              wavefrontHitQueueA.UAVBarrier(cmdList);
              wavefrontHitQueueB.UAVBarrier(cmdList);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }
          prepareWavefrontDispatchArgs(PrepareAfterTraceProfileNames[bounce]);

          uint32 hitReadQueue = currentQueue;
          // Only a global sort produces a new hit ordering. Without it, and for
          // block-local sorting, shade can consume the hit queue in place.
          const bool globalSortThisBounce = g_wavefront_reorder &&
                                             g_wavefront_block_sort == false &&
                                             (bounce > 0 || g_wavefront_skip_primary_sort == false);
          const uint32 hitWriteQueue = globalSortThisBounce ? hitReadQueue ^ 1 : hitReadQueue;
          bindWavefrontConstants(bounce, hitReadQueue, hitWriteQueue);

          if(gpuWavefrontPath || globalSortThisBounce)
          {
              ProfileBlock sortPB(cmdList, HitSortProfileNames[bounce]);
              cmdList->SetPipelineState(wavefrontClearReorderPSO);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       hitMetaDispatchArgsOffset, nullptr, 0);
              wavefrontCounterBuffer.UAVBarrier(cmdList);

              cmdList->SetPipelineState(wavefrontCountReorderBinsPSOVariants[wavefrontVariantIdx]);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       sortHitDispatchArgsOffset, nullptr, 0);
              wavefrontCounterBuffer.UAVBarrier(cmdList);

              cmdList->SetPipelineState(wavefrontPrefixReorderBinsPSO);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       hitMetaDispatchArgsOffset, nullptr, 0);
              wavefrontCounterBuffer.UAVBarrier(cmdList);

              cmdList->SetPipelineState(wavefrontScatterReorderedRaysPSOVariants[wavefrontVariantIdx]);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       gpuWavefrontPath ? scatterHitDispatchArgsOffset : hitDispatchArgsOffset, nullptr, 0);
              if(globalSortThisBounce)
              {
                  wavefrontHitQueueA.UAVBarrier(cmdList);
                  wavefrontHitQueueB.UAVBarrier(cmdList);
                  wavefrontCounterBuffer.UAVBarrier(cmdList);
              }
          }

          bindWavefrontConstants(bounce, hitWriteQueue, nextQueue);
          {
              ProfileBlock shadePB(cmdList, ShadeHitProfileNames[bounce]);
              cmdList->SetPipelineState(wavefrontShadeHitsPSOVariants[wavefrontVariantIdx]);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       hitDispatchArgsOffset, nullptr, 0);
              wavefrontPathStateBuffer.UAVBarrier(cmdList);
              wavefrontRayQueueA.UAVBarrier(cmdList);
              wavefrontRayQueueB.UAVBarrier(cmdList);
              wavefrontShadowQueue.UAVBarrier(cmdList);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }
          prepareWavefrontDispatchArgs(PrepareAfterShadeProfileNames[bounce]);

          {
              ProfileBlock shadowPB(cmdList, TraceShadowProfileNames[bounce]);
              cmdList->SetPipelineState(wavefrontTraceShadowsPSOVariants[wavefrontVariantIdx]);
              cmdList->ExecuteIndirect(wavefrontDispatchCommandSignature, 1, wavefrontDispatchArgsBuffer.Resource(),
                                       shadowDispatchArgsOffset, nullptr, 0);
              wavefrontPathStateBuffer.UAVBarrier(cmdList);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }

          currentQueue = nextQueue;

          {
              ProfileBlock advancePB(cmdList, AdvanceProfileNames[bounce]);
              cmdList->SetPipelineState(wavefrontAdvancePSO);
              DX12::CmdList->Dispatch(1, 1, 1);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }
      }

      bindWavefrontConstants(0, 0, 1);
      {
          ProfileBlock accumulatePB(cmdList, "WF Accumulate");
          cmdList->SetPipelineState(wavefrontAccumulatePSO);
          DX12::CmdList->Dispatch(gx, gy, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          rtTarget.UAVBarrier(cmdList);
      }
      break;
    }
