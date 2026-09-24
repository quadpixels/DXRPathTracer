    case 7: {
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
      const uint32 wavefrontThreadGroupSize = ActiveWavefrontThreadGroupSize();
      const uint32 pixelGroups = (width * height + wavefrontThreadGroupSize - 1) / wavefrontThreadGroupSize;
      const uint32 persistentGroups = Clamp<uint32>(uint32(g_persistent_worker_groups_actual), 1, Max<uint32>(pixelGroups, 1));
      const uint32 gx = (width + 7) / 8;
      const uint32 gy = (height + 7) / 8;

      rtConstants.myFlags &= ~(2u | 4u | 16u | 128u | 256u | 512u);
      if(g_persistent_tiled)
      {
          rtConstants.myFlags |= 128u;
          if(g_persistent_tiled_order == 1)
              rtConstants.myFlags |= 256u | 512u;
          else if(g_persistent_tiled_order == 2)
              rtConstants.myFlags |= 512u;
      }
      rtConstants.WavefrontReadQueue = 0;
      rtConstants.WavefrontWriteQueue = 1;
      rtConstants.WavefrontBounce = 0;
      rtConstants.WavefrontPadding = Clamp<uint32>(uint32(g_persistent_batch_waves), 1, 8);
      rtConstants.WavefrontThreadGroupSize = wavefrontThreadGroupSize;
      rtConstants.PersistentWorkerCount = persistentGroups * wavefrontThreadGroupSize;
      DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);

      clearWavefrontCounters();

      {
          ProfileBlock generatePB(cmdList, "Persistent Wavefront Generate Primary");
          cmdList->SetPipelineState(wavefrontGeneratePrimaryPSO);
          DX12::CmdList->Dispatch(gx, gy, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          wavefrontRayQueueA.UAVBarrier(cmdList);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
      }

      uint32 currentQueue = 0;
      for(uint32 bounce = 0; bounce < uint32(AppSettings::MaxPathLength); ++bounce)
      {
          const uint32 nextQueue = currentQueue ^ 1;
          rtConstants.WavefrontReadQueue = currentQueue;
          rtConstants.WavefrontWriteQueue = nextQueue;
          rtConstants.WavefrontBounce = bounce;
          DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);

          {
              ProfileBlock preparePB(cmdList, "Persistent Wavefront Prepare Rays");
              cmdList->SetPipelineState(wavefrontPreparePersistentBouncePSO);
              DX12::CmdList->Dispatch(1, 1, 1);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }

          {
              ProfileBlock tracePB(cmdList, "Persistent Wavefront Trace And Shade");
              cmdList->SetPipelineState(wavefrontPersistentTraceShadePSOVariants[wavefrontVariantIdx]);
              DX12::CmdList->Dispatch(persistentGroups, 1, 1);
              wavefrontPathStateBuffer.UAVBarrier(cmdList);
              wavefrontRayQueueA.UAVBarrier(cmdList);
              wavefrontRayQueueB.UAVBarrier(cmdList);
              wavefrontShadowQueue.UAVBarrier(cmdList);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }

          {
              ProfileBlock preparePB(cmdList, "Persistent Wavefront Prepare Shadows");
              cmdList->SetPipelineState(wavefrontPreparePersistentBouncePSO);
              DX12::CmdList->Dispatch(1, 1, 1);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }

          {
              ProfileBlock shadowPB(cmdList, "Persistent Wavefront Trace Shadows");
              cmdList->SetPipelineState(wavefrontPersistentTraceShadowsPSOVariants[wavefrontVariantIdx]);
              DX12::CmdList->Dispatch(persistentGroups, 1, 1);
              wavefrontPathStateBuffer.UAVBarrier(cmdList);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }

          {
              ProfileBlock advancePB(cmdList, "Persistent Wavefront Advance");
              cmdList->SetPipelineState(wavefrontAdvancePSO);
              DX12::CmdList->Dispatch(1, 1, 1);
              wavefrontCounterBuffer.UAVBarrier(cmdList);
          }

          currentQueue = nextQueue;
      }

      {
          ProfileBlock accumulatePB(cmdList, "Persistent Wavefront Accumulate");
          cmdList->SetPipelineState(wavefrontAccumulatePSO);
          DX12::CmdList->Dispatch(gx, gy, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          rtTarget.UAVBarrier(cmdList);
      }
      break;
    }
