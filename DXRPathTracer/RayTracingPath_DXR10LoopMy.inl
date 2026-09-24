    case 3: {
      ProfileBlock pb(cmdList, "TraceRay DispatchRays (loop, my)");
      cmdList->SetPipelineState1(rtPSOLoop_my);
      D3D12_DISPATCH_RAYS_DESC dispatchDesc = {};
      dispatchDesc.HitGroupTable = rtHitTableLoop_my.ShaderTable();
      dispatchDesc.MissShaderTable = rtMissTableLoop_my.ShaderTable();
      dispatchDesc.RayGenerationShaderRecord = rtRayGenTableLoop_my.ShaderRecord(0);
      dispatchDesc.Width = uint32(rtTarget.Width());
      dispatchDesc.Height = uint32(rtTarget.Height());
      dispatchDesc.Depth = 1;
      DX12::CmdList->DispatchRays(&dispatchDesc);
      break;
      break;
    }
