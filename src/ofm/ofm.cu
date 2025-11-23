#include "ofm.h"
#include "ofm_util.h"
#include <cub/cub.cuh>

namespace ofm {
OFM::OFM(int3 _tile_dim)
{
    Alloc(_tile_dim);
}

void OFM::Alloc(int3 _tile_dim)
{
    tile_dim_     = _tile_dim;

    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    int voxel_num   = Prod(tile_dim_) * 512;
    int x_voxel_num = Prod(x_tile_dim) * 512;
    int y_voxel_num = Prod(y_tile_dim) * 512;
    int z_voxel_num = Prod(z_tile_dim) * 512;

    // boundary
    is_bc_x_  = std::make_shared<DHMemory<uint8_t>>(x_voxel_num);
    is_bc_y_  = std::make_shared<DHMemory<uint8_t>>(y_voxel_num);
    is_bc_z_  = std::make_shared<DHMemory<uint8_t>>(z_voxel_num);
    bc_val_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    bc_val_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    bc_val_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);

    // backward flow map
    T_x_   = std::make_shared<DHMemory<float3>>(x_voxel_num);
    T_y_   = std::make_shared<DHMemory<float3>>(y_voxel_num);
    T_z_   = std::make_shared<DHMemory<float3>>(z_voxel_num);
    psi_x_ = std::make_shared<DHMemory<float3>>(x_voxel_num);
    psi_y_ = std::make_shared<DHMemory<float3>>(y_voxel_num);
    psi_z_ = std::make_shared<DHMemory<float3>>(z_voxel_num);

    // forward flow map
    F_x_   = std::make_shared<DHMemory<float3>>(x_voxel_num);
    F_y_   = std::make_shared<DHMemory<float3>>(y_voxel_num);
    F_z_   = std::make_shared<DHMemory<float3>>(z_voxel_num);
    phi_x_ = std::make_shared<DHMemory<float3>>(x_voxel_num);
    phi_y_ = std::make_shared<DHMemory<float3>>(y_voxel_num);
    phi_z_ = std::make_shared<DHMemory<float3>>(z_voxel_num);

    // velocity storage
    u_        = std::make_shared<DHMemory<float3>>(voxel_num);
    u_x_      = std::make_shared<DHMemory<float>>(x_voxel_num);
    u_y_      = std::make_shared<DHMemory<float>>(y_voxel_num);
    u_z_      = std::make_shared<DHMemory<float>>(z_voxel_num);
    init_u_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    init_u_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    init_u_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);
    mid_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    mid_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    mid_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);
    tmp_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    tmp_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    tmp_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);
    err_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    err_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    err_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);

    // vorticity
    vor_norm_ = std::make_shared<DHMemory<float>>(voxel_num);

    // solver
    int min_dim   = tile_dim_.x;
    min_dim       = min_dim > tile_dim_.y ? tile_dim_.y : min_dim;
    min_dim       = min_dim > tile_dim_.z ? tile_dim_.z : min_dim;
    int level_num = (int)log2(min_dim) + 1;
    amgpcg_.Alloc(_tile_dim, level_num);
}

void OFM::SetProfilier(GPUTimer* _profiler) {
    profiler_ = _profiler;
}


void OFM::UpdateBoundary(cudaStream_t _stream)
{
    if (use_dynamic_solid_) {
        {
            CUDA_PROFILE_SCOPE(*profiler_, _stream, "UpdateBoundaryCondition");
            SetBcBySurfaceAsync(*is_bc_x_, *is_bc_y_, *is_bc_z_, *bc_val_x_, *bc_val_y_, *bc_val_z_, tile_dim_, voxel_tex_, velocity_tex_, voxelized_velocity_scaler_, _stream);
            SetCoefByIsBcAsync(*(amgpcg_.poisson_vector_[0].is_dof_), *(amgpcg_.poisson_vector_[0].a_diag_), *(amgpcg_.poisson_vector_[0].a_x_), *(amgpcg_.poisson_vector_[0].a_y_),
                                       *(amgpcg_.poisson_vector_[0].a_z_), tile_dim_, *is_bc_x_, *is_bc_y_, *is_bc_z_, _stream);
        }

        {
            CUDA_PROFILE_SCOPE(*profiler_, _stream, "Rebuild Projection Matrix")
            amgpcg_.BuildAsync(6.0f, -1.0f, _stream);
        }
    }
}

void OFM::AdvanceAsync(float _dt, cudaStream_t _stream)
{
    float mid_dt = 0.5f * _dt;
    std::shared_ptr<DHMemory<float>> last_proj_u_x = init_u_x_;
    std::shared_ptr<DHMemory<float>> last_proj_u_y = init_u_y_;
    std::shared_ptr<DHMemory<float>> last_proj_u_z = init_u_z_;
    std::shared_ptr<DHMemory<float>> src_u_x = init_u_x_;
    std::shared_ptr<DHMemory<float>> src_u_y = init_u_y_;
    std::shared_ptr<DHMemory<float>> src_u_z = init_u_z_;

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Advection");
        AdvectN2XAsync(*tmp_u_x_, tile_dim_, *src_u_x, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
        AdvectN2YAsync(*tmp_u_y_, tile_dim_, *src_u_y, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
        AdvectN2ZAsync(*tmp_u_z_, tile_dim_, *src_u_z, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
    }

    SetInletAsync(*bc_val_x_, *bc_val_y_, tile_dim_, inlet_norm_, inlet_angle_, _stream);

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Projection 1");
        ProjectAsync(_stream);
    }

    mid_u_x_.swap(tmp_u_x_);
    mid_u_y_.swap(tmp_u_y_);
    mid_u_z_.swap(tmp_u_z_);

    step_++;
}

void OFM::ReinitAsync(float _dt, cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };

    ResetForwardFlowMapAsync(_stream);
    ResetBackwardFlowMapAsync(_stream);

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Marching Backward flowmap");
        RKAxisAsync(*psi_x_, *T_x_, tile_dim_, x_tile_dim, *mid_u_x_, *mid_u_y_, *mid_u_z_, grid_origin_, dx_, _dt, _stream);
        RKAxisAsync(*psi_y_, *T_y_, tile_dim_, y_tile_dim, *mid_u_x_, *mid_u_y_, *mid_u_z_, grid_origin_, dx_, _dt, _stream);
        RKAxisAsync(*psi_z_, *T_z_, tile_dim_, z_tile_dim, *mid_u_x_, *mid_u_y_, *mid_u_z_, grid_origin_, dx_, _dt, _stream);
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Marching Forward flowmap");
        RKAxisAsync(*phi_x_, *F_x_, tile_dim_, x_tile_dim, *mid_u_x_, *mid_u_y_, *mid_u_z_, grid_origin_, dx_, -_dt, _stream);
        RKAxisAsync(*phi_y_, *F_y_, tile_dim_, y_tile_dim, *mid_u_x_, *mid_u_y_, *mid_u_z_, grid_origin_, dx_, -_dt, _stream);
        RKAxisAsync(*phi_z_, *F_z_, tile_dim_, z_tile_dim, *mid_u_x_, *mid_u_y_, *mid_u_z_, grid_origin_, dx_, -_dt, _stream);
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Impulse reconstruction");
        PullbackAxisAsync(*u_x_, tile_dim_, x_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_x_, *T_x_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*u_y_, tile_dim_, y_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_y_, *T_y_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*u_z_, tile_dim_, z_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_z_, *T_z_, grid_origin_, dx_, _stream);
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "BFECC");
        PullbackAxisAsync(*err_u_x_, tile_dim_, x_tile_dim, *u_x_, *u_y_, *u_z_, *phi_x_, *F_x_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*err_u_y_, tile_dim_, y_tile_dim, *u_x_, *u_y_, *u_z_, *phi_y_, *F_y_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*err_u_z_, tile_dim_, z_tile_dim, *u_x_, *u_y_, *u_z_, *phi_z_, *F_z_, grid_origin_, dx_, _stream);
        AddFieldsAsync(*err_u_x_, x_tile_dim, *err_u_x_, *init_u_x_, -1.0f, _stream);
        AddFieldsAsync(*err_u_y_, y_tile_dim, *err_u_y_, *init_u_y_, -1.0f, _stream);
        AddFieldsAsync(*err_u_z_, z_tile_dim, *err_u_z_, *init_u_z_, -1.0f, _stream);
        PullbackAxisAsync(*init_u_x_, tile_dim_, x_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_x_, *T_x_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*init_u_y_, tile_dim_, y_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_y_, *T_y_, grid_origin_, dx_, _stream);
        PullbackAxisAsync(*init_u_z_, tile_dim_, z_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_z_, *T_z_, grid_origin_, dx_, _stream);
        AddFieldsAsync(*tmp_u_x_, x_tile_dim, *u_x_, *init_u_x_, -0.5f, _stream);
        AddFieldsAsync(*tmp_u_y_, y_tile_dim, *u_y_, *init_u_y_, -0.5f, _stream);
        AddFieldsAsync(*tmp_u_z_, z_tile_dim, *u_z_, *init_u_z_, -0.5f, _stream);
        if (use_bfecc_clamp_) {
            int3 x_max_ijk = { tile_dim_.x * 8, tile_dim_.y * 8 - 1, tile_dim_.z * 8 - 1 };
            int3 y_max_ijk = { tile_dim_.x * 8 - 1, tile_dim_.y * 8, tile_dim_.z * 8 - 1 };
            int3 z_max_ijk = { tile_dim_.x * 8 - 1, tile_dim_.y * 8 - 1, tile_dim_.z * 8 };
            BfeccClampAsync(*tmp_u_x_, x_tile_dim, x_max_ijk, *u_x_, _stream);
            BfeccClampAsync(*tmp_u_y_, y_tile_dim, y_max_ijk, *u_y_, _stream);
            BfeccClampAsync(*tmp_u_z_, z_tile_dim, z_max_ijk, *u_z_, _stream);
        }
    }

    {
        CUDA_PROFILE_SCOPE(*profiler_, _stream, "Projection 2");
        ProjectAsync(_stream);
    }

    init_u_x_.swap(tmp_u_x_);
    init_u_y_.swap(tmp_u_y_);
    init_u_z_.swap(tmp_u_z_);
}

void OFM::ResetForwardFlowMapAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetToIdentityXASync(*phi_x_, *F_x_, x_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityYASync(*phi_y_, *F_y_, y_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityZASync(*phi_z_, *F_z_, z_tile_dim, grid_origin_, dx_, _stream);
}

void OFM::ResetBackwardFlowMapAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetToIdentityXASync(*psi_x_, *T_x_, x_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityYASync(*psi_y_, *T_y_, y_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityZASync(*psi_z_, *T_z_, z_tile_dim, grid_origin_, dx_, _stream);
}

void OFM::ProjectAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };

    SetBcAxisAsync(*tmp_u_x_, x_tile_dim, *is_bc_x_, *bc_val_x_, _stream);
    SetBcAxisAsync(*tmp_u_y_, y_tile_dim, *is_bc_y_, *bc_val_y_, _stream);
    SetBcAxisAsync(*tmp_u_z_, z_tile_dim, *is_bc_z_, *bc_val_z_, _stream);

    CalcDivAsync(*(amgpcg_.b_), tile_dim_, *(amgpcg_.poisson_vector_[0].is_dof_), *tmp_u_x_, *tmp_u_y_, *tmp_u_z_, _stream);

    amgpcg_.SolveAsync(_stream);

    ApplyPressureAsync(*tmp_u_x_, *tmp_u_y_, *tmp_u_z_, tile_dim_, *(amgpcg_.x_), *is_bc_x_, *is_bc_y_, *is_bc_z_, _stream);
}
}
