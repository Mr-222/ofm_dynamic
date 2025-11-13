#pragma once

#include "mem.h"

namespace lfm {
void GetCenteralVecAsync(DHMemory<float3>& _vec, int3 _tile_dim, const DHMemory<float>& _vec_x, const DHMemory<float>& _vec_y, const DHMemory<float>& _vec_z, cudaStream_t _stream);

void GetVorNormAsync(DHMemory<float>& _vor_norm, int3 _tile_dim, const DHMemory<float3>& _u, float _dx, cudaStream_t _stream);

void ResetToIdentityXASync(DHMemory<float3>& _psi_x, DHMemory<float3>& _T_x, int3 _x_tile_dim, float3 _grid_origin, float _dx, cudaStream_t _stream);

void ResetToIdentityYASync(DHMemory<float3>& _psi_y, DHMemory<float3>& _T_y, int3 _y_tile_dim, float3 _grid_origin, float _dx, cudaStream_t _stream);

void ResetToIdentityZASync(DHMemory<float3>& _psi_z, DHMemory<float3>& _T_z, int3 _z_tile_dim, float3 _grid_origin, float _dx, cudaStream_t _stream);

void RKAxisAsync(DHMemory<float3>& _psi_axis, DHMemory<float3>& _T_axis, int3 _tile_dim, int3 _axis_tile_dim,
                 const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, float3 _grid_origin, float _dx, float _dt, cudaStream_t _stream);

void PullbackAxisAsync(DHMemory<float>& _dst_axis, int3 _tile_dim, int3 _axis_tile_dim, const DHMemory<float>& _src_x, const DHMemory<float>& _src_y, const DHMemory<float>& _src_z,
                       const DHMemory<float3>& _psi_axis, const DHMemory<float3>& _T_axis, float3 _grid_origin, float _dx, cudaStream_t _stream);

void PullbackCenterAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float3>& _psi_c, float3 _grid_origin, float _dx, cudaStream_t _stream);

void AddFieldsAsync(DHMemory<float>& _dst, int3 _tile_dim, DHMemory<float>& _src1, DHMemory<float>& _src2, float _coef2, cudaStream_t _stream);

void GetCentralPsiAsync(DHMemory<float3>& _psi_c, int3 _tile_dim, const DHMemory<float3>& _psi_x, const DHMemory<float3>& _psi_y, const DHMemory<float3>& _psi_z, cudaStream_t _stream);

void BfeccClampAsync(DHMemory<float>& _after_bfecc, int3 _tile_dim, int3 _max_ijk, const DHMemory<float>& _before_bfecc, cudaStream_t _stream);

void RKAxisAccumulateForceAsync(int _rk_order, DHMemory<float3>& _psi_axis, DHMemory<float3>& _T_axis, DHMemory<float>& _f_axis, int3 _tile_dim, int3 _axis_tile_dim,
                                const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, const DHMemory<float>& _f_x, const DHMemory<float>& _f_y, const DHMemory<float>& _f_z, float3 _grid_origin, float _dx, float _dt, cudaStream_t _stream);

void SetBcAxisAsync(DHMemory<float>& _u_axis, int3 _axis_tile_dim, const DHMemory<uint8_t>& _is_bc_axis, const DHMemory<float>& _bc_val_axis, cudaStream_t _stream);

void CalcDivAsync(DHMemory<float>& _b, int3 _tile_dim, const DHMemory<uint8_t>& _is_dof, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, cudaStream_t _stream);

void ApplyPressureAsync(DHMemory<float>& _u_x, DHMemory<float>& _u_y, DHMemory<float>& _u_z, int3 _tile_dim, const DHMemory<float>& _p, const DHMemory<uint8_t>& _is_bc_x, const DHMemory<uint8_t>& _is_bc_y, const DHMemory<uint8_t>& _is_bc_z, cudaStream_t _stream);

void SetWallBcAsync(DHMemory<uint8_t>& _is_bc_x, DHMemory<uint8_t>& _is_bc_y, DHMemory<uint8_t>& _is_bc_z, DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, DHMemory<float>& _bc_val_z, int3 _tile_dim,
                    float3 _neg_bc_val, float3 _pos_bc_val, cudaStream_t _stream);

void SetBcByPhiAsync(DHMemory<uint8_t>& _is_bc_x, DHMemory<uint8_t>& _is_bc_y, DHMemory<uint8_t>& _is_bc_z, DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, DHMemory<float>& _bc_val_z, int3 _tile_dim, const DHMemory<float>& _phi, cudaStream_t _stream);

void SetBcBySurfaceAsync(DHMemory<uint8_t>& _is_bc_x, DHMemory<uint8_t>& _is_bc_y, DHMemory<uint8_t>& _is_bc_z, DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, DHMemory<float>& _bc_val_z, int3 _tile_dim, const cudaSurfaceObject_t& voxel_surface, const cudaSurfaceObject_t& velocity_surface, float vel_scaler, cudaStream_t _stream);

void SetCoefByIsBcAsync(DHMemory<uint8_t>& _is_dof, DHMemory<float>& _a_diag, DHMemory<float>& _a_x, DHMemory<float>& _a_y, DHMemory<float>& _a_z, int3 _tile_dim, const DHMemory<uint8_t>& _is_bc_x, const DHMemory<uint8_t>& _is_bc_y, const DHMemory<uint8_t>& _is_bc_z, cudaStream_t _stream);

void AdvectN2XAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream);

void AdvectN2YAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream);

void AdvectN2ZAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream);

void AdvectN2CAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream);

void SetInletAsync(DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, int3 _tile_dim, float _inlet_norm, float _inlet_angle, cudaStream_t _stream);
}