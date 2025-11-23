add_rules("mode.release", "mode.debug")

includes("./../src/engine/xmake.lua")
includes("./../src/ofm/xmake.lua")

add_requires("vulkansdk", "glfw 3.4", "glm 1.0.1")
add_requires("glslang 1.3", { configs = { binaryonly = true } })
add_requires("imgui 1.91.1",  {configs = {glfw_vulkan = true}})
add_requires("cuda", {system=true, configs={utils={"cublas","cusparse","cusolver"}}})
add_requires("vtk 9.3.1")

set_policy("build.intermediate_directory", false)
set_runtimes("MD")

includes("sim_render", "voxelization", "dynamic_obstacle")
add_options("compile_commands")

option("all")
    set_default(true)
    set_showmenu(false)
    set_description("Build all examples")