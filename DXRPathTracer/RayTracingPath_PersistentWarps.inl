    case 8:
    case 14: {
      const bool staticStridePersistentWarps = activeRenderPath == 14;
      ProfileBlock pb(cmdList, staticStridePersistentWarps ? "Persistent Warps (Static Stride) Dispatch" : "Persistent Warps (Atomic Cursor) Dispatch");

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
      const uint32 numPixels = width * height;
      const uint32 wavefrontThreadGroupSize = ActiveWavefrontThreadGroupSize();
      const uint32 pixelGroups = (numPixels + wavefrontThreadGroupSize - 1) / wavefrontThreadGroupSize;
      const uint32 persistentGroups = Clamp<uint32>(uint32(g_persistent_worker_groups_actual), 1, Max<uint32>(pixelGroups, 1));

      rtConstants.myFlags &= ~(2u | 4u | 128u | 256u | 512u);
      if(g_persistent_tiled)
      {
          rtConstants.myFlags |= 128u;
          if(g_persistent_tiled_order == 1)
              rtConstants.myFlags |= 256u | 512u;
          else if(g_persistent_tiled_order == 2)
              rtConstants.myFlags |= 512u;
      }
      rtConstants.WavefrontBounce = 0;
      rtConstants.WavefrontReadQueue = 0;
      rtConstants.WavefrontWriteQueue = 1;
      rtConstants.WavefrontPadding = Clamp<uint32>(uint32(g_persistent_batch_waves), 1, 8);
      rtConstants.PersistentWorkerCount = persistentGroups * wavefrontThreadGroupSize;
      DX12::BindTempConstantBuffer(cmdList, rtConstants, RTParams_CBuffer, CmdListMode::Compute);

      if(staticStridePersistentWarps == false)
      {
          clearWavefrontCounters();

          ProfileBlock preparePB(cmdList, "Persistent Warps Prepare");
          cmdList->SetPipelineState(wavefrontPreparePersistentBouncePSO);
          DX12::CmdList->Dispatch(1, 1, 1);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
      }

      {
          ProfileBlock tracePB(cmdList, staticStridePersistentWarps ? "Persistent Warps Path Trace (Static Stride)" : "Persistent Warps Path Trace (Atomic Cursor)");
          cmdList->SetPipelineState(persistentWarpsPathTracePSOVariants[wavefrontVariantIdx]);
          DX12::CmdList->Dispatch(persistentGroups, 1, 1);
          wavefrontCounterBuffer.UAVBarrier(cmdList);
          rtTarget.UAVBarrier(cmdList);
      }
      break;
    }

