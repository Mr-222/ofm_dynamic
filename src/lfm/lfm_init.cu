#include "data_io.h"
#include "lfm_init.h"
#include "lfm_util.h"
namespace ofm {
void InitOFMAsync(OFM& _ofm, const OFMConfiguration& _config, cudaStream_t _stream)
{
    // alloc
    int3 tile_dim    = { _config.tile_dim[0], _config.tile_dim[1], _config.tile_dim[2] };
    _ofm.Alloc(tile_dim);

    // simulation parameter
    _ofm.step_     = 0;

    // domain
    _ofm.dx_            = _config.len_y / (8 * tile_dim.y);
    _ofm.grid_origin_.x = _config.grid_origin[0];
    _ofm.grid_origin_.y = _config.grid_origin[1];
    _ofm.grid_origin_.z = _config.grid_origin[2];

    // boundary
    _ofm.inlet_norm_   = _config.inlet_norm;
    _ofm.inlet_angle_  = _config.inlet_angle;
    float pi           = 3.1415926f;
    float radian_angle = _ofm.inlet_angle_ / 180.0f * pi;
    float3 neg_bc_val  = { _ofm.inlet_norm_ * cos(radian_angle), _ofm.inlet_norm_ * sin(radian_angle), 0.0f };
    float3 pos_bc_val  = neg_bc_val;
    SetWallBcAsync(*_ofm.is_bc_x_, *_ofm.is_bc_y_, *_ofm.is_bc_z_, *_ofm.bc_val_x_, *_ofm.bc_val_y_, *_ofm.bc_val_z_, tile_dim, neg_bc_val, pos_bc_val, _stream);

    // static boundary
    bool use_static_solid = _config.use_static_solid;
    DHMemory<float> solid_sdf(Prod(tile_dim) * 512);

    if (use_static_solid) {
        DHMemory<float> solid_sdf_np(Prod(tile_dim) * 512);
        ReadNpy<float>(_config.solid_sdf_path, solid_sdf_np.host_ptr_);
        solid_sdf_np.HostToDevAsync(_stream);
        ConToTileAsync(solid_sdf, tile_dim, solid_sdf_np, _stream);
        SetBcByPhiAsync(*_ofm.is_bc_x_, *_ofm.is_bc_y_, *_ofm.is_bc_z_, *_ofm.bc_val_x_, *_ofm.bc_val_y_, *_ofm.bc_val_z_, tile_dim, solid_sdf, _stream);
    }

    _ofm.use_dynamic_solid_ = _config.use_dynamic_solid;

    // poisson
    {
        SetCoefByIsBcAsync(*(_ofm.amgpcg_.poisson_vector_[0].is_dof_), *(_ofm.amgpcg_.poisson_vector_[0].a_diag_), *(_ofm.amgpcg_.poisson_vector_[0].a_x_), *(_ofm.amgpcg_.poisson_vector_[0].a_y_),
                           *(_ofm.amgpcg_.poisson_vector_[0].a_z_), tile_dim, *_ofm.is_bc_x_, *_ofm.is_bc_y_, *_ofm.is_bc_z_, _stream);
        _ofm.amgpcg_.BuildAsync(6.0f, -1.0f, _stream);
        _ofm.amgpcg_.solve_by_tol_ = false;
        _ofm.amgpcg_.max_iter_ = 6;
    }

    // bfecc clamp
    _ofm.use_bfecc_clamp_ = _config.use_bfecc_clamp;
}
}