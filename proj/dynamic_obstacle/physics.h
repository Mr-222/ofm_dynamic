#pragma once

#include "function/physics/cuda_engine.h"
#include "lfm.h"
#include "timer.h"

class PhysicsEngineUser : public CudaEngine {
    lfm::LFM lfm_;
    GPUTimer profiler_;
    virtual void initExternalMem() override;

public:
    virtual void init(Configuration& config, GlobalContext* g_ctx) override;
    virtual void step() override;
    virtual void cleanup() override;
};
