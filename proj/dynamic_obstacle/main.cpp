#include "engine.h"
#include "function/render/render_engine.h"
#include "function/physics/cuda_engine.h"
#include "ui.h"

int main()
{
    RenderEngine render_engine;
    UIEngineUser ui_engine;
    CudaEngine physics_engine;

    Configuration config = load("./config/dynamic_obstacle.json");
    Engine engine;
    engine.init(config, &render_engine, &ui_engine, &physics_engine);
    engine.run();
    engine.cleanup();

    return 0;
}
