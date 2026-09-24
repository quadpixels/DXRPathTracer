    case 10: {
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

      // Clear all queue counters in one command before profiling the path.
      clearWavefrontCounters();

      if(g_wavefront_use_clear_uav == false)
      {
          ProfileBlock cursorPB(cmdList, "Wavefront Work Cursor Clear (Kernel)");
          cmdList->SetPipelineState(wavefrontPreparePersistentBouncePSO);
          DX12::CmdList->Dispatch(1, 1, 1);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
      }

      ProfileBlock pb(cmdList, "RayQuery Persistent Wavefront Global Queue Dispatch");

      const uint32 width = uint32(rtTarget.Width());
      const uint32 height = uint32(rtTarget.Height());
      const uint32 numPixels = width * height;
      const uint32 wavefrontThreadGroupSize = ActiveWavefrontThreadGroupSize();
      const uint32 pixelGroups = (numPixels + wavefrontThreadGroupSize - 1) / wavefrontThreadGroupSize;
      const uint32 persistentGroups = Clamp<uint32>(uint32(g_persistent_worker_groups), 1, Max<uint32>(pixelGroups, 1));
      const uint32 gx = (width + 7) / 8;
      const uint32 gy = (height + 7) / 8;

      rtConstants.myFlags = 16u;
      rtConstants.WavefrontReadQueue = 0;
      rtConstants.WavefrontWriteQueue = 0;
      rtConstants.WavefrontBounce = 0;
      rtConstants.WavefrontPadding = Clamp<uint32>(uint32(g_persistent_batch_waves), 1, 8);
      rtConstants.WavefrontThreadGroupSize = wavefrontThreadGroupSize;
      DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);

      {
          ProfileBlock generatePB(cmdList, "Persistent Global Queue Generate Primary");
          cmdList->SetPipelineState(wavefrontGeneratePrimaryPSO);
          DX12::CmdList->Dispatch(gx, gy, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          wavefrontRayQueueA.UAVBarrier(cmdList);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
      }

      {
          ProfileBlock workPB(cmdList, "Persistent Global Queue Workers");
          cmdList->SetPipelineState(wavefrontPersistentWorkQueuePSOVariants[wavefrontVariantIdx]);
          DX12::CmdList->Dispatch(persistentGroups, 1, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          wavefrontRayQueueA.UAVBarrier(cmdList);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
      }

      {
          ProfileBlock accumulatePB(cmdList, "Persistent Global Queue Accumulate");
          cmdList->SetPipelineState(wavefrontAccumulatePSO);
          DX12::CmdList->Dispatch(gx, gy, 1);
          wavefrontPathStateBuffer.UAVBarrier(cmdList);
          rtTarget.UAVBarrier(cmdList);
      }
      break;
    }
