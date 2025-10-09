#include "engine.h"
#include "function/render/render_engine.h"
#include "physics.h"
#include "ui.h"

int main()
{
    RenderEngine render_engine;
    UIEngineUser ui_engine;
    PhysicsEngineUser physics_engine;

    Configuration config = load("./config/dynamic_obstacle.json");
    Engine engine;
    engine.init(config, &render_engine, &ui_engine, &physics_engine);
    engine.run();
    engine.cleanup();

    return 0;
}
