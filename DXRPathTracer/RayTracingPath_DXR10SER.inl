    case 1: {
      ProfileBlock pb(cmdList, "TraceRay DispatchRays (recursive, SER)");
      cmdList->SetPipelineState1(rtPSO_SER);
      D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
      dispatchDesc.HitGroupTable = rtHitTable_SER.ShaderTable();
      dispatchDesc.MissShaderTable = rtMissTable_SER.ShaderTable();
      dispatchDesc.RayGenerationShaderRecord = rtRayGenTable_SER.ShaderRecord(0);
      dispatchDesc.Width = uint32(rtTarget.Width());
      dispatchDesc.Height = uint32(rtTarget.Height());
      dispatchDesc.Depth = 1;
      DX12::CmdList->DispatchRays(&dispatchDesc);
      break;
      break;
    }
