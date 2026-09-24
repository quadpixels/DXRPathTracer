    case 12: {
      ProfileBlock pb(cmdList, "RayQuery Persistent Warps (RayGen)");
      cmdList->SetPipelineState1(rtRayQueryPersistentPSO);
      D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
      dispatchDesc.RayGenerationShaderRecord = rtRayQueryPersistentRayGenTable.ShaderRecord(0);
      dispatchDesc.Width = ActiveWavefrontThreadGroupSize();
      dispatchDesc.Height = uint32(Max<int>(g_persistent_worker_groups, 1));
      dispatchDesc.Depth = 1;
      DX12::CmdList->DispatchRays(&dispatchDesc);
      break;
    }
