    case 2: {
      ProfileBlock pb(cmdList, "TraceRay DispatchRays (loop, SER)");
      cmdList->SetPipelineState1(rtPSOLoop_SER);
      D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
      dispatchDesc.HitGroupTable = rtHitTableLoop_SER.ShaderTable();
      dispatchDesc.MissShaderTable = rtMissTableLoop_SER.ShaderTable();
      dispatchDesc.RayGenerationShaderRecord = rtRayGenTableLoop_SER.ShaderRecord(0);
      dispatchDesc.Width = uint32(rtTarget.Width());
      dispatchDesc.Height = uint32(rtTarget.Height());
      dispatchDesc.Depth = 1;
      DX12::CmdList->DispatchRays(&dispatchDesc);
      break;
      break;
    }
