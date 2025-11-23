add_rules("mode.release", "mode.debug")

target("voxelization")
    set_targetdir("build")

    if is_plat("windows") then
        add_rules("plugin.vsxmake.autoupdate")
        add_cxxflags("/utf-8")
    end
    add_rules("utils.glsl2spv", { outputdir = "build" })

    set_languages("cxx20")
    set_kind("binary")

    add_headerfiles("*.h")
    add_files("*.cpp")
    add_files("*.cu")
    add_includedirs(".",{public=true})

    add_cugencodes("compute_75")
    add_cuflags("--std c++20", "-lineinfo")

    add_deps("engine")
    add_deps("ofm")

    add_packages("imgui")
    add_packages("vulkansdk", "glfw", "glm")
    add_packages("cuda")
    add_packages("vtk")

    if is_mode("debug") then
        add_cxxflags("-DDEBUG")
    end
    if is_mode("release") then
        add_cxxflags("-DNDEBUG")
    end
