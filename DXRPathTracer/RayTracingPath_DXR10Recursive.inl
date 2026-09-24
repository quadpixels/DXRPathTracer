    case 0: {
      ProfileBlock pb(cmdList, "TraceRay DispatchRays (original, recursive)");
      cmdList->SetPipelineState1(rtPSO);
      D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
      dispatchDesc.HitGroupTable = rtHitTable.ShaderTable();
      dispatchDesc.MissShaderTable = rtMissTable.ShaderTable();
      dispatchDesc.RayGenerationShaderRecord = rtRayGenTable.ShaderRecord(0);
      dispatchDesc.Width = uint32(rtTarget.Width());
      dispatchDesc.Height = uint32(rtTarget.Height());
      dispatchDesc.Depth = 1;
      DX12::CmdList->DispatchRays(&dispatchDesc);
      break;
    }
