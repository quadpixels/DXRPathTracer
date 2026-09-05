//=================================================================================================
//
//  MJP's DX12 Sample Framework
//  http://mynameismjp.wordpress.com/
//
//  All code licensed under the MIT license
//
//=================================================================================================

#include "PCH.h"
#include "Profiler.h"
#include "DX12.h"
#include "..\\Utility.h"
#include "..\\ImGui\ImGui.h"
#include <string>

using std::wstring;
using std::map;
using std::string;

extern bool g_wavefront_reorder;
extern bool g_wavefront_skip_primary_sort;
extern bool g_wavefront_block_sort;
extern bool g_wavefront_wave_append;
extern bool g_persistent_shadow_workers;
extern int g_persistent_worker_groups;
extern int g_persistent_batch_waves;

namespace SampleFramework12
{

// == Profiler ====================================================================================

Profiler Profiler::GlobalProfiler;

static const uint64 MaxProfiles = 128;

struct ProfileData
{
    const char* Name = nullptr;

    bool QueryStarted = false;
    bool QueryFinished = false ;
    bool Active = false;

    bool CPUProfile = false;
    int64 StartTime = 0;
    int64 EndTime = 0;

    static const uint64 FilterSize = 64;
    double TimeSamples[FilterSize] = { };
    uint64 CurrSample = 0;
};

void Profiler::Initialize()
{
    Shutdown();

    enableGPUProfiling = true;

    D3D12_QUERY_HEAP_DESC heapDesc = { };
    heapDesc.Count = MaxProfiles * 2;
    heapDesc.NodeMask = 0;
    heapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    DX12::Device->CreateQueryHeap(&heapDesc, IID_PPV_ARGS(&queryHeap));

    readbackBuffer.Initialize(MaxProfiles * DX12::RenderLatency * 2 * sizeof(uint64));
    readbackBuffer.Resource->SetName(L"Query Readback Buffer");

    profiles.Init(MaxProfiles);
    cpuProfiles.Init(MaxProfiles);
}

void Profiler::Shutdown()
{
    DX12::DeferredRelease(queryHeap);
    readbackBuffer.Shutdown();
    profiles.Shutdown();
    cpuProfiles.Shutdown();
    numProfiles = 0;
}

uint64 Profiler::StartProfile(ID3D12GraphicsCommandList* cmdList, const char* name)
{
    Assert_(name != nullptr);
    if(enableGPUProfiling == false)
        return uint64(-1);

    uint64 profileIdx = uint64(-1);
    for(uint64 i = 0; i < numProfiles; ++i)
    {
        if(profiles[i].Name == name)
        {
            profileIdx = i;
            break;
        }
    }

    if(profileIdx == uint64(-1))
    {
        Assert_(numProfiles < MaxProfiles);
        profileIdx = numProfiles++;
        profiles[profileIdx].Name = name;
    }

    ProfileData& profileData = profiles[profileIdx];
    Assert_(profileData.QueryStarted == false);
    Assert_(profileData.QueryFinished == false);
    profileData.CPUProfile = false;
    profileData.Active = true;

    // Insert the start timestamp
    const uint32 startQueryIdx = uint32(profileIdx * 2);
    cmdList->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, startQueryIdx);

    profileData.QueryStarted = true;

    return profileIdx;
}

void Profiler::EndProfile(ID3D12GraphicsCommandList* cmdList, uint64 idx)
{
    if(enableGPUProfiling == false)
        return;

    Assert_(idx < numProfiles);

    ProfileData& profileData = profiles[idx];
    Assert_(profileData.QueryStarted == true);
    Assert_(profileData.QueryFinished == false);

    // Insert the end timestamp
    const uint32 startQueryIdx = uint32(idx * 2);
    const uint32 endQueryIdx = startQueryIdx + 1;
    cmdList->EndQuery(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, endQueryIdx);

    // Resolve the data
    const uint64 dstOffset = ((DX12::CurrFrameIdx * MaxProfiles * 2) + startQueryIdx) * sizeof(uint64);
    cmdList->ResolveQueryData(queryHeap, D3D12_QUERY_TYPE_TIMESTAMP, startQueryIdx, 2, readbackBuffer.Resource, dstOffset);

    profileData.QueryStarted = false;
    profileData.QueryFinished = true;
}

uint64 Profiler::StartCPUProfile(const char* name)
{
    Assert_(name != nullptr);

    uint64 profileIdx = uint64(-1);
    for(uint64 i = 0; i < numCPUProfiles; ++i)
    {
        if(cpuProfiles[i].Name == name)
        {
            profileIdx = i;
            break;
        }
    }

    if(profileIdx == uint64(-1))
    {
        Assert_(numCPUProfiles < MaxProfiles);
        profileIdx = numCPUProfiles++;
        cpuProfiles[profileIdx].Name = name;
    }

    ProfileData& profileData = cpuProfiles[profileIdx];
    Assert_(profileData.QueryStarted == false);
    Assert_(profileData.QueryFinished == false);
    profileData.CPUProfile = true;
    profileData.Active = true;

    timer.Update();
    profileData.StartTime = timer.ElapsedMicroseconds();

    profileData.QueryStarted = true;

    return profileIdx;
}

void Profiler::EndCPUProfile(uint64 idx)
{
    Assert_(idx < numCPUProfiles);

    ProfileData& profileData = cpuProfiles[idx];
    Assert_(profileData.QueryStarted == true);
    Assert_(profileData.QueryFinished == false);

    timer.Update();
    profileData.EndTime = timer.ElapsedMicroseconds();

    profileData.QueryStarted = false;
    profileData.QueryFinished = true;
}

static void UpdateProfile(ProfileData& profile, uint64 profileIdx, bool drawText, uint64 gpuFrequency, const uint64* frameQueryData)
{
    profile.QueryFinished = false;

    double time = 0.0f;
    if(profile.CPUProfile)
    {
        time = double(profile.EndTime - profile.StartTime) / 1000.0;
    }
    else if(frameQueryData)
    {
        Assert_(frameQueryData != nullptr);

        // Get the query data
        uint64 startTime = frameQueryData[profileIdx * 2 + 0];
        uint64 endTime = frameQueryData[profileIdx * 2 + 1];

        if(endTime > startTime)
        {
            uint64 delta = endTime - startTime;
            double frequency = double(gpuFrequency);
            time = (delta / frequency) * 1000.0;
        }
    }

    profile.TimeSamples[profile.CurrSample] = time;
    profile.CurrSample = (profile.CurrSample + 1) % ProfileData::FilterSize;

    double maxTime = 0.0;
    double avgTime = 0.0;
    uint64 avgTimeSamples = 0;
    for(UINT i = 0; i < ProfileData::FilterSize; ++i)
    {
        if(profile.TimeSamples[i] <= 0.0)
            continue;
        maxTime = Max(profile.TimeSamples[i], maxTime);
        avgTime += profile.TimeSamples[i];
        ++avgTimeSamples;
    }

    if(avgTimeSamples > 0)
        avgTime /= double(avgTimeSamples);

    if(profile.Active && drawText)
        ImGui::Text("%s: %.2fms (%.2fms max)", profile.Name, avgTime, maxTime);

    profile.Active = false;
}

static double AverageProfileTime(const ProfileData& profile)
{
    double avgTime = 0.0;
    uint64 avgTimeSamples = 0;
    for(UINT i = 0; i < ProfileData::FilterSize; ++i)
    {
        if(profile.TimeSamples[i] <= 0.0)
            continue;

        avgTime += profile.TimeSamples[i];
        ++avgTimeSamples;
    }

    if(avgTimeSamples == 0)
        return 0.0;

    return avgTime / double(avgTimeSamples);
}

static double ProfileTimeByName(const Array<ProfileData>& profiles, uint64 numProfiles, const char* name)
{
    for(uint64 i = 0; i < numProfiles; ++i)
    {
        if(profiles[i].Name != nullptr && strcmp(profiles[i].Name, name) == 0)
            return AverageProfileTime(profiles[i]);
    }

    return 0.0;
}

static void AppendTimingLine(string& text, const char* label, double time)
{
    char line[256] = { };
    sprintf_s(line, "%-32s %.3f ms\r\n", label, time);
    text += line;
}

static string BuildWavefrontTimingSummary(const Array<ProfileData>& profiles, uint64 numProfiles)
{
    static const char* TraceHitProfileNames[] =
    {
        "WF Trace Hits B0", "WF Trace Hits B1", "WF Trace Hits B2", "WF Trace Hits B3",
        "WF Trace Hits B4", "WF Trace Hits B5", "WF Trace Hits B6", "WF Trace Hits B7",
    };

    static const char* HitSortProfileNames[] =
    {
        "WF Hit Sort B0", "WF Hit Sort B1", "WF Hit Sort B2", "WF Hit Sort B3",
        "WF Hit Sort B4", "WF Hit Sort B5", "WF Hit Sort B6", "WF Hit Sort B7",
    };

    static const char* ShadeHitProfileNames[] =
    {
        "WF Shade Hits B0", "WF Shade Hits B1", "WF Shade Hits B2", "WF Shade Hits B3",
        "WF Shade Hits B4", "WF Shade Hits B5", "WF Shade Hits B6", "WF Shade Hits B7",
    };

    static const char* TraceShadowProfileNames[] =
    {
        "WF Trace Shadows B0", "WF Trace Shadows B1", "WF Trace Shadows B2", "WF Trace Shadows B3",
        "WF Trace Shadows B4", "WF Trace Shadows B5", "WF Trace Shadows B6", "WF Trace Shadows B7",
    };

    static const char* AdvanceProfileNames[] =
    {
        "WF Advance B0", "WF Advance B1", "WF Advance B2", "WF Advance B3",
        "WF Advance B4", "WF Advance B5", "WF Advance B6", "WF Advance B7",
    };

    static const char* PrepareAfterTraceProfileNames[] =
    {
        "WF Prepare Args After Trace B0", "WF Prepare Args After Trace B1", "WF Prepare Args After Trace B2", "WF Prepare Args After Trace B3",
        "WF Prepare Args After Trace B4", "WF Prepare Args After Trace B5", "WF Prepare Args After Trace B6", "WF Prepare Args After Trace B7",
    };

    static const char* PrepareAfterShadeProfileNames[] =
    {
        "WF Prepare Args After Shade B0", "WF Prepare Args After Shade B1", "WF Prepare Args After Shade B2", "WF Prepare Args After Shade B3",
        "WF Prepare Args After Shade B4", "WF Prepare Args After Shade B5", "WF Prepare Args After Shade B6", "WF Prepare Args After Shade B7",
    };

    static const char* PersistentPrepareProfileNames[] =
    {
        "Persistent Prepare B0", "Persistent Prepare B1", "Persistent Prepare B2", "Persistent Prepare B3",
        "Persistent Prepare B4", "Persistent Prepare B5", "Persistent Prepare B6", "Persistent Prepare B7",
    };

    static const char* PersistentTraceShadeProfileNames[] =
    {
        "Persistent Trace+Shade B0", "Persistent Trace+Shade B1", "Persistent Trace+Shade B2", "Persistent Trace+Shade B3",
        "Persistent Trace+Shade B4", "Persistent Trace+Shade B5", "Persistent Trace+Shade B6", "Persistent Trace+Shade B7",
    };

    static const char* PersistentPrepareArgsProfileNames[] =
    {
        "Persistent Prepare Args B0", "Persistent Prepare Args B1", "Persistent Prepare Args B2", "Persistent Prepare Args B3",
        "Persistent Prepare Args B4", "Persistent Prepare Args B5", "Persistent Prepare Args B6", "Persistent Prepare Args B7",
    };

    static const char* PersistentTraceShadowProfileNames[] =
    {
        "Persistent Trace Shadows B0", "Persistent Trace Shadows B1", "Persistent Trace Shadows B2", "Persistent Trace Shadows B3",
        "Persistent Trace Shadows B4", "Persistent Trace Shadows B5", "Persistent Trace Shadows B6", "Persistent Trace Shadows B7",
    };

    static const char* PersistentAdvanceProfileNames[] =
    {
        "Persistent Advance B0", "Persistent Advance B1", "Persistent Advance B2", "Persistent Advance B3",
        "Persistent Advance B4", "Persistent Advance B5", "Persistent Advance B6", "Persistent Advance B7",
    };

    double traceTotal = 0.0;
    double sortTotal = 0.0;
    double shadeTotal = 0.0;
    double shadowTotal = 0.0;
    double advanceTotal = 0.0;
    double prepareTotal = ProfileTimeByName(profiles, numProfiles, "WF Prepare Args Initial");
    const char* sortMode = "None";
    if(g_wavefront_reorder)
        sortMode = g_wavefront_block_sort ? "Thread Block" : "Global";

    for(uint64 bounce = 0; bounce < ArraySize_(TraceHitProfileNames); ++bounce)
    {
        const bool includeGlobalSort = g_wavefront_reorder && g_wavefront_block_sort == false &&
                                       (bounce > 0 || g_wavefront_skip_primary_sort == false);

        traceTotal += ProfileTimeByName(profiles, numProfiles, TraceHitProfileNames[bounce]);
        if(includeGlobalSort)
            sortTotal += ProfileTimeByName(profiles, numProfiles, HitSortProfileNames[bounce]);
        shadeTotal += ProfileTimeByName(profiles, numProfiles, ShadeHitProfileNames[bounce]);
        shadowTotal += ProfileTimeByName(profiles, numProfiles, TraceShadowProfileNames[bounce]);
        advanceTotal += ProfileTimeByName(profiles, numProfiles, AdvanceProfileNames[bounce]);
        prepareTotal += ProfileTimeByName(profiles, numProfiles, PrepareAfterTraceProfileNames[bounce]);
        prepareTotal += ProfileTimeByName(profiles, numProfiles, PrepareAfterShadeProfileNames[bounce]);
    }

    string text;
    text += "Wavefront GPU timing summary\r\n";
    text += "============================\r\n";
    text += "Sort Mode: ";
    text += sortMode;
    text += g_wavefront_skip_primary_sort ? " (skip B0)\r\n" : " (include B0)\r\n";
    text += "Append Mode: ";
    text += g_wavefront_wave_append ? "Wave aggregated atomics\r\n" : "Per-item atomics\r\n";
    AppendTimingLine(text, "RayQuery Wavefront Dispatch", ProfileTimeByName(profiles, numProfiles, "RayQuery Wavefront Dispatch"));
    AppendTimingLine(text, "WF Clear", ProfileTimeByName(profiles, numProfiles, "WF Clear"));
    AppendTimingLine(text, "WF Generate Primary", ProfileTimeByName(profiles, numProfiles, "WF Generate Primary"));
    AppendTimingLine(text, "WF Trace Hits Total", traceTotal);
    AppendTimingLine(text, "WF Global Hit Sort Total", sortTotal);
    AppendTimingLine(text, "WF Shade Hits Total", shadeTotal);
    AppendTimingLine(text, "WF Trace Shadows Total", shadowTotal);
    AppendTimingLine(text, "WF Prepare Args Total", prepareTotal);
    AppendTimingLine(text, "WF Advance Total", advanceTotal);
    AppendTimingLine(text, "WF Accumulate", ProfileTimeByName(profiles, numProfiles, "WF Accumulate"));
    text += "\r\nPer-bounce detail\r\n";

    for(uint64 bounce = 0; bounce < ArraySize_(TraceHitProfileNames); ++bounce)
    {
        char label[64] = { };
        sprintf_s(label, "B%llu Trace Hits", bounce);
        AppendTimingLine(text, label, ProfileTimeByName(profiles, numProfiles, TraceHitProfileNames[bounce]));

        sprintf_s(label, "B%llu Hit Sort", bounce);
        if(g_wavefront_reorder && g_wavefront_block_sort == false &&
           (bounce > 0 || g_wavefront_skip_primary_sort == false))
            AppendTimingLine(text, label, ProfileTimeByName(profiles, numProfiles, HitSortProfileNames[bounce]));
        else
            AppendTimingLine(text, label, 0.0);

        sprintf_s(label, "B%llu Shade Hits", bounce);
        AppendTimingLine(text, label, ProfileTimeByName(profiles, numProfiles, ShadeHitProfileNames[bounce]));

        sprintf_s(label, "B%llu Trace Shadows", bounce);
        AppendTimingLine(text, label, ProfileTimeByName(profiles, numProfiles, TraceShadowProfileNames[bounce]));
    }

    const double persistentTotal = ProfileTimeByName(profiles, numProfiles, "RayQuery Persistent Workers Dispatch");
    if(persistentTotal > 0.0)
    {
        double persistentPrepareTotal = 0.0;
        double persistentTraceShadeTotal = 0.0;
        double persistentPrepareArgsTotal = 0.0;
        double persistentShadowTotal = 0.0;
        double persistentAdvanceTotal = 0.0;

        for(uint64 bounce = 0; bounce < ArraySize_(PersistentTraceShadeProfileNames); ++bounce)
        {
            persistentPrepareTotal += ProfileTimeByName(profiles, numProfiles, PersistentPrepareProfileNames[bounce]);
            persistentTraceShadeTotal += ProfileTimeByName(profiles, numProfiles, PersistentTraceShadeProfileNames[bounce]);
            persistentPrepareArgsTotal += ProfileTimeByName(profiles, numProfiles, PersistentPrepareArgsProfileNames[bounce]);
            persistentShadowTotal += ProfileTimeByName(profiles, numProfiles, PersistentTraceShadowProfileNames[bounce]);
            persistentAdvanceTotal += ProfileTimeByName(profiles, numProfiles, PersistentAdvanceProfileNames[bounce]);
        }

        text += "\r\nPersistent GPU timing summary\r\n";
        text += "=============================\r\n";
        char settingsLine[256] = { };
        sprintf_s(settingsLine, "Persistent Settings: worker groups=%d, batch waves=%d, shadow=%s\r\n",
                  g_persistent_worker_groups, g_persistent_batch_waves,
                  g_persistent_shadow_workers ? "persistent workers" : "indirect dispatch");
        text += settingsLine;
        AppendTimingLine(text, "RayQuery Persistent Workers", persistentTotal);
        AppendTimingLine(text, "Persistent Clear", ProfileTimeByName(profiles, numProfiles, "Persistent Clear"));
        AppendTimingLine(text, "Persistent Generate Primary", ProfileTimeByName(profiles, numProfiles, "Persistent Generate Primary"));
        AppendTimingLine(text, "Persistent Prepare Total", persistentPrepareTotal);
        AppendTimingLine(text, "Persistent Trace+Shade Total", persistentTraceShadeTotal);
        AppendTimingLine(text, "Persistent Prepare Args Total", persistentPrepareArgsTotal);
        AppendTimingLine(text, "Persistent Trace Shadows Total", persistentShadowTotal);
        AppendTimingLine(text, "Persistent Advance Total", persistentAdvanceTotal);
        AppendTimingLine(text, "Persistent Accumulate", ProfileTimeByName(profiles, numProfiles, "Persistent Accumulate"));
        text += "\r\nPersistent per-bounce detail\r\n";

        for(uint64 bounce = 0; bounce < ArraySize_(PersistentTraceShadeProfileNames); ++bounce)
        {
            char label[64] = { };
            sprintf_s(label, "B%llu Persistent Trace+Shade", bounce);
            AppendTimingLine(text, label, ProfileTimeByName(profiles, numProfiles, PersistentTraceShadeProfileNames[bounce]));

            sprintf_s(label, "B%llu Persistent Shadows", bounce);
            AppendTimingLine(text, label, ProfileTimeByName(profiles, numProfiles, PersistentTraceShadowProfileNames[bounce]));
        }
    }

    return text;
}

void Profiler::EndFrame(uint32 displayWidth, uint32 displayHeight)
{
    uint64 gpuFrequency = 0;
    const uint64* frameQueryData = nullptr;
    if(enableGPUProfiling)
    {
        DX12::GfxQueue->GetTimestampFrequency(&gpuFrequency);

        const uint64* queryData = readbackBuffer.Map<uint64>();
        frameQueryData = queryData + (DX12::CurrFrameIdx * MaxProfiles * 2);
    }

    bool drawText = false;
    if(showUI == false)
    {
        ImGui::SetNextWindowSize(ImVec2(75.0f, 25.0f));
        ImGui::SetNextWindowPos(ImVec2(25.0f, 50.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar;
        if(ImGui::Begin("profiler_button", nullptr, ImVec2(75.0f, 25.0f), 0.0f, flags))
        {
            if(ImGui::Button("Timing"))
                showUI = true;
        }

        ImGui::PopStyleVar();
    }
    else
    {
        ImVec2 initialSize = ImVec2(displayWidth * 0.5f, float(displayHeight) * 0.25f);
        ImGui::SetNextWindowSize(initialSize, ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiSetCond_FirstUseEver);

        drawText = ImGui::Begin("Timing", &showUI);

        if(logToClipboard)
            ImGui::LogToClipboard();
    }

    if(drawText)
    {
        ImGui::Text("GPU Timing");
        ImGui::Separator();
    }

    // Iterate over all of the profiles
    for(uint64 profileIdx = 0; profileIdx < numProfiles; ++profileIdx)
        UpdateProfile(profiles[profileIdx], profileIdx, drawText, gpuFrequency, frameQueryData);

    if(drawText)
    {
        ImGui::Text(" ");
        ImGui::Text("CPU Timing");
        ImGui::Separator();
    }

    for(uint64 profileIdx = 0; profileIdx < numCPUProfiles; ++profileIdx)
        UpdateProfile(cpuProfiles[profileIdx], profileIdx, drawText, gpuFrequency, frameQueryData);

    if(showUI)
    {
        if(logToClipboard)
            ImGui::LogFinish();

        ImGui::Text(" ");
        logToClipboard = ImGui::Button("Copy To Clipboard");
        ImGui::SameLine();
        if(ImGui::Button("Copy Wavefront Summary"))
        {
            const string summary = BuildWavefrontTimingSummary(profiles, numProfiles);
            ImGui::SetClipboardText(summary.c_str());
        }
    }
    else
        logToClipboard = false;

    ImGui::End();

    if(enableGPUProfiling)
        readbackBuffer.Unmap();

    enableGPUProfiling = showUI;
}

double Profiler::GPUProfileTiming(const char* name) const
{
    uint64 profileIdx = uint64(-1);
    for(uint64 i = 0; i < numProfiles; ++i)
    {
        if(profiles[i].Name == name)
        {
            profileIdx = i;
            break;
        }
    }

    if(profileIdx == uint64(-1))
        return 0.0;

    uint64 gpuFrequency = 0;
    DX12::GfxQueue->GetTimestampFrequency(&gpuFrequency);

    const uint64* queryData = readbackBuffer.Map<uint64>();
    const uint64* frameQueryData = queryData + (DX12::CurrFrameIdx * MaxProfiles * 2);

    // Get the query data
    uint64 startTime = frameQueryData[profileIdx * 2 + 0];
    uint64 endTime = frameQueryData[profileIdx * 2 + 1];

    double time = 0.0;
    if(endTime > startTime)
    {
        uint64 delta = endTime - startTime;
        double frequency = double(gpuFrequency);
        time = (delta / frequency) * 1000.0;
    }

    readbackBuffer.Unmap();

    return time;
}

// == ProfileBlock ================================================================================

ProfileBlock::ProfileBlock(ID3D12GraphicsCommandList* cmdList_, const char* name) : cmdList(cmdList_)
{
    idx = Profiler::GlobalProfiler.StartProfile(cmdList, name);
}

ProfileBlock::~ProfileBlock()
{
    Profiler::GlobalProfiler.EndProfile(cmdList, idx);
}

// == CPUProfileBlock =============================================================================

CPUProfileBlock::CPUProfileBlock(const char* name)
{
    idx = Profiler::GlobalProfiler.StartCPUProfile(name);
}

CPUProfileBlock::~CPUProfileBlock()
{
    Profiler::GlobalProfiler.EndCPUProfile(idx);
}

}
