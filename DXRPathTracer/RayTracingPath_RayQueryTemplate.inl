    case 4: {
      ProfileBlock pb(cmdList, "RayQuery Dispatch (template)");
      cmdList->SetPipelineState(rtRayQueryPSO);
      uint32_t gx, gy;
      gx = static_cast<uint32_t>(rtTarget.Width() - 1) / 8 + 1;
      gy = static_cast<uint32_t>(rtTarget.Height() - 1) / 8 + 1;
      DX12::CmdList->Dispatch(gx, gy, 1);
      break;
    }
