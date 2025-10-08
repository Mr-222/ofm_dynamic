#pragma once

#include "amgpcg.h"

namespace lfm {
class LFM {
public:
    // domain
    int3 tile_dim_;
    float dx_;
    float3 grid_origin_;

    // simulation parameters
    int rk_order_ = 4;
    int step_;

    // boundary
    float inlet_norm_;
    float inlet_angle_;
    std::shared_ptr<DHMemory<uint8_t>> is_bc_x_;
    std::shared_ptr<DHMemory<uint8_t>> is_bc_y_;
    std::shared_ptr<DHMemory<uint8_t>> is_bc_z_;
    std::shared_ptr<DHMemory<float>> bc_val_x_;
    std::shared_ptr<DHMemory<float>> bc_val_y_;
    std::shared_ptr<DHMemory<float>> bc_val_z_;

    // backward flow map
    std::shared_ptr<DHMemory<float3>> T_x_;
    std::shared_ptr<DHMemory<float3>> T_y_;
    std::shared_ptr<DHMemory<float3>> T_z_;
    std::shared_ptr<DHMemory<float3>> psi_x_;
    std::shared_ptr<DHMemory<float3>> psi_y_;
    std::shared_ptr<DHMemory<float3>> psi_z_;

    /// forward flow map
    std::shared_ptr<DHMemory<float3>> F_x_;
    std::shared_ptr<DHMemory<float3>> F_y_;
    std::shared_ptr<DHMemory<float3>> F_z_;
    std::shared_ptr<DHMemory<float3>> phi_x_;
    std::shared_ptr<DHMemory<float3>> phi_y_;
    std::shared_ptr<DHMemory<float3>> phi_z_;

    // velocity storage
    std::shared_ptr<DHMemory<float3>> u_;
    std::shared_ptr<DHMemory<float>> u_x_;
    std::shared_ptr<DHMemory<float>> u_y_;
    std::shared_ptr<DHMemory<float>> u_z_;
    std::shared_ptr<DHMemory<float>> init_u_x_;
    std::shared_ptr<DHMemory<float>> init_u_y_;
    std::shared_ptr<DHMemory<float>> init_u_z_;
    std::shared_ptr<DHMemory<float>> tmp_u_x_;
    std::shared_ptr<DHMemory<float>> tmp_u_y_;
    std::shared_ptr<DHMemory<float>> tmp_u_z_;
    std::shared_ptr<DHMemory<float>> err_u_x_;
    std::shared_ptr<DHMemory<float>> err_u_y_;
    std::shared_ptr<DHMemory<float>> err_u_z_;
    std::shared_ptr<DHMemory<float>> mid_u_x_;
    std::shared_ptr<DHMemory<float>> mid_u_y_;
    std::shared_ptr<DHMemory<float>> mid_u_z_;

    // vorticity
    std::shared_ptr<DHMemory<float>> vor_norm_;

    // solver
    AMGPCG amgpcg_;

    // bfecc clamp
    bool use_bfecc_clamp_;

    LFM() = default;
    LFM(int3 _tile_dim);
    void Alloc(int3 _tile_dim);

    void AdvanceAsync(float _dt, cudaStream_t _stream);
    void ReinitAsync(float _dt, cudaStream_t _stream);
    void ResetForwardFlowMapAsync(cudaStream_t _stream);
    void ResetBackwardFlowMapAsync(cudaStream_t _stream);
    void ProjectAsync(cudaStream_t _stream);
};
}