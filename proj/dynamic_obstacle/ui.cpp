#include "ui.h"
#include "core/config/config.h"
#include "core/tool/logger.h"
#include "function/global_context.h"
#include "function/resource_manager/resource_manager.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include "imgui_stdlib.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

std::string UIEngineUser::record_output_path;
bool UIEngineUser::is_input_text_focused = false;

void UIEngineUser::init(const Configuration& config, void* render_to_ui)
{
    ImGuiEngine::init(config, render_to_ui);
    prev_mouse_delta = ImVec2(0, 0);

    record_output_path = config.at("recorder").at("output_path");
}

void UIEngineUser::handleInput()
{
    handleMouseInput();
    handleKeyboardInput();
}

void UIEngineUser::handleMouseInput()
{
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
        ImVec2 mouse_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
        float dx           = mouse_delta.x - prev_mouse_delta.x;
        float dy           = mouse_delta.y - prev_mouse_delta.y;
        g_ctx.rm->camera.update_rotation(dx, dy);

        prev_mouse_delta = mouse_delta;
    } else {
        prev_mouse_delta = ImVec2(0, 0);
    }
}

void UIEngineUser::handleKeyboardInput()
{
    if (is_input_text_focused) {
        return;
    }

    if (ImGui::IsKeyDown(ImGuiKey_W))
        g_ctx.rm->camera.go_forward();
    if (ImGui::IsKeyDown(ImGuiKey_S))
        g_ctx.rm->camera.go_backward();
    if (ImGui::IsKeyDown(ImGuiKey_A))
        g_ctx.rm->camera.go_left();
    if (ImGui::IsKeyDown(ImGuiKey_D))
        g_ctx.rm->camera.go_right();
    if (ImGui::IsKeyDown(ImGuiKey_E))
        g_ctx.rm->camera.go_up();
    if (ImGui::IsKeyDown(ImGuiKey_Q))
        g_ctx.rm->camera.go_down();
}

void UIEngineUser::recordingUI()
{
    ImGui::Begin("Recording");

    ImGui::Text("Record Output Path");
    ImGui::SameLine();
    ImGui::InputText("##Record Output Path", &record_output_path);
    if (ImGui::IsItemActive()) {
        is_input_text_focused = true;
    } else {
        is_input_text_focused = false;
    }

    if (g_ctx.rm->recorder.is_recording) {
        if (ImGui::Button("Stop Recording")) {
            g_ctx.rm->recorder.end();
        }
    } else {
        if (ImGui::Button("StartRecording")) {
            g_ctx.rm->recorder.begin(
                record_output_path,
                g_ctx.vk.swapChainImages[0]->extent.width,
                g_ctx.vk.swapChainImages[0]->extent.height);
        }
    }
    ImGui::End();
}

std::function<void(VkCommandBuffer)> UIEngineUser::getDrawUIFunction()
{
    return [](VkCommandBuffer commandBuffer) {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ImGui::Begin("Materials");
        // for (auto& material_pair : g_ctx.rm->materials) {
        //     auto& material = material_pair.second;
        //     ImGui::Text("%s", material.name.c_str());
        //     ImGui::Text("\tmaterial");
        //     if (ImGui::ColorEdit3(
        //             (std::string("color##") + material.name).c_str(),
        //             glm::value_ptr(material.data.color))) {
        //         material.update(material.data);
        //     }
        //     if (ImGui::SliderFloat(
        //             (std::string("roughness##") + material.name).c_str(),
        //             &material.data.roughness, 0, 1)) {
        //         material.update(material.data);
        //     }
        //     if (ImGui::SliderFloat(
        //             (std::string("metallic##") + material.name).c_str(),
        //             &material.data.metallic, 0, 1)) {
        //         material.update(material.data);
        //     }
        //     ImGui::Text("");
        // }
        // ImGui::End();

        ImGui::Begin("Camera");
        auto& camera = g_ctx.rm->camera;
        if (ImGui::DragFloat3("position", glm::value_ptr(camera.data.eye_w), 0.03f)) {
            camera.update_position(camera.data.eye_w);
        }
        if (ImGui::DragFloat("yaw", &camera.rotation.x, 1.0f, -180.0f, 180.0f)) {
            camera.update_rotation(camera.rotation);
        }
        if (ImGui::DragFloat("pitch", &camera.rotation.y, 1.0f, -89.0f, 89.0f)) {
            camera.update_rotation(camera.rotation);
        }
        if (ImGui::DragFloat("fov", &camera.data.fov_y, 1, 30, 120)) {
            camera.update_fov(camera.data.fov_y);
        }
        // if (ImGui::Button("Print camera parameters")) {
        //     INFO_ALL("Camera position: {}, {}, {}", camera.data.eye_w.x, camera.data.eye_w.y, camera.data.eye_w.z);
        //     INFO_ALL("Camera rotation: {}, {}, {}", camera.data.view_dir.x, camera.data.view_dir.y, camera.data.view_dir.z);
        //}
        ImGui::End();

        // ImGui::Begin("Lights");
        // auto& lights = g_ctx.rm->lights;
        // for (int i = 0; i < lights.data.size(); i++) {
        //     auto& light = lights.data[i];
        //     ImGui::Text("%d", i);
        //     if (ImGui::DragFloat3(
        //             (std::string("posOrDir##") + std::to_string(i)).c_str(),
        //             glm::value_ptr(light.posOrDir), 0.03f)) {
        //         lights.update(&light, i, 1);
        //     }
        //     if (ImGui::DragFloat3(
        //             (std::string("intensity##") + std::to_string(i)).c_str(),
        //             glm::value_ptr(light.intensity), 0.03f, 0, FLT_MAX)) {
        //         lights.update(&light, i, 1);
        //     }
        // }
        // ImGui::End();

        // recordingUI();

        drawAxis();

        ImGui::Render();
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    };
}
