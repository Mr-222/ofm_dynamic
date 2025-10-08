#pragma once

#include "function/ui/imgui_engine.h"
#include "imgui.h"
#include <string>

class UIEngineUser : public ImGuiEngine {
    void handleMouseInput();
    void handleKeyboardInput();
    static void recordingUI();

    ImVec2 prev_mouse_delta;
    static std::string record_output_path;
    static bool is_input_text_focused;

public:
    virtual void handleInput() override;
    virtual std::function<void(VkCommandBuffer)> getDrawUIFunction() override;
    virtual void init(const Configuration& config, void* render_to_ui) override;
};
