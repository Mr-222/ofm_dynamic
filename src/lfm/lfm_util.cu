#include "lfm_util.h"
#include "util.h"
#include <iostream>

namespace lfm {
__global__ void GetCentralVecKernel(float3* _vec, int3 _tile_dim, const float* _vec_x, const float* _vec_y, const float* _vec_z)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);

    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };

    int t_id = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int3 right_ijk = { ijk.x + 1, ijk.y, ijk.z };
        int3 up_ijk    = { ijk.x, ijk.y + 1, ijk.z };
        int3 front_ijk = { ijk.x, ijk.y, ijk.z + 1 };
        _vec[idx].x    = (_vec_x[IjkToIdx(x_tile_dim, ijk)] + _vec_x[IjkToIdx(x_tile_dim, right_ijk)]) * 0.5f;
        _vec[idx].y    = (_vec_y[IjkToIdx(y_tile_dim, ijk)] + _vec_y[IjkToIdx(y_tile_dim, up_ijk)]) * 0.5f;
        _vec[idx].z    = (_vec_z[IjkToIdx(z_tile_dim, ijk)] + _vec_z[IjkToIdx(z_tile_dim, front_ijk)]) * 0.5f;
    }
}

void GetCenteralVecAsync(DHMemory<float3>& _vec, int3 _tile_dim, const DHMemory<float>& _vec_x, const DHMemory<float>& _vec_y, const DHMemory<float>& _vec_z, cudaStream_t _stream)
{
    float3* vec  = _vec.dev_ptr_;
    float* vec_x = _vec_x.dev_ptr_;
    float* vec_y = _vec_y.dev_ptr_;
    float* vec_z = _vec_z.dev_ptr_;
    int tile_num = Prod(_tile_dim);
    GetCentralVecKernel<<<tile_num, 128, 0, _stream>>>(vec, _tile_dim, vec_x, vec_y, vec_z);
}

__global__ void GetVorNormKernel(float* _vor_norm, int3 _tile_dim, const float3* _u, float _inv_dist)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float3 vor, vr, vl, vt, vb, vc, va;
        if (ijk.x < 8 * _tile_dim.x - 1) {
            int3 right_ijk = { ijk.x + 1, ijk.y, ijk.z };
            vr             = _u[IjkToIdx(_tile_dim, right_ijk)];
        } else
            vr = _u[IjkToIdx(_tile_dim, ijk)];
        if (ijk.x > 0) {
            int3 left_ijk = { ijk.x - 1, ijk.y, ijk.z };
            vl            = _u[IjkToIdx(_tile_dim, left_ijk)];
        } else
            vl = _u[IjkToIdx(_tile_dim, ijk)];
        if (ijk.y < 8 * _tile_dim.y - 1) {
            int3 up_ijk = { ijk.x, ijk.y + 1, ijk.z };
            vt          = _u[IjkToIdx(_tile_dim, up_ijk)];
        } else
            vt = _u[IjkToIdx(_tile_dim, ijk)];
        if (ijk.y > 0) {
            int3 bottom_ijk = { ijk.x, ijk.y - 1, ijk.z };
            vb              = _u[IjkToIdx(_tile_dim, bottom_ijk)];
        } else
            vb = _u[IjkToIdx(_tile_dim, ijk)];
        if (ijk.z < 8 * _tile_dim.z - 1) {
            int3 front_ijk = { ijk.x, ijk.y, ijk.z + 1 };
            vc             = _u[IjkToIdx(_tile_dim, front_ijk)];
        } else
            vc = _u[IjkToIdx(_tile_dim, ijk)];
        if (ijk.z > 0) {
            int3 back_ijk = { ijk.x, ijk.y, ijk.z - 1 };
            va            = _u[IjkToIdx(_tile_dim, back_ijk)];
        } else
            va = _u[IjkToIdx(_tile_dim, ijk)];
        vor.x             = _inv_dist * (vt.z - vb.z - vc.y + va.y);
        vor.y             = _inv_dist * (vc.x - va.x - vr.z + vl.z);
        vor.z             = _inv_dist * (vr.y - vl.y - vt.x + vb.x);
        int np_idx        = ijk.x * _tile_dim.y * _tile_dim.z * 64 + ijk.y * _tile_dim.z * 8 + ijk.z;
        _vor_norm[np_idx] = sqrtf(vor.x * vor.x + vor.y * vor.y + vor.z * vor.z);
    }
}

void GetVorNormAsync(DHMemory<float>& _vor_norm, int3 _tile_dim, const DHMemory<float3>& _u, float _dx, cudaStream_t _stream)
{
    float* vor_norm = _vor_norm.dev_ptr_;
    float3* u       = _u.dev_ptr_;
    int tile_num    = Prod(_tile_dim);
    float inv_dist  = 1.0f / (2.0f * _dx);
    GetVorNormKernel<<<tile_num, 128, 0, _stream>>>(vor_norm, _tile_dim, u, inv_dist);
}

__global__ void ResetToIdentityXKernel(float3* _psi_x, float3* _T_x, int3 _x_tile_dim, float3 _grid_origin, float _dx)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_x_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int idx        = tile_idx * 512 + voxel_idx;
        float3 pos     = { _grid_origin.x + ijk.x * _dx, _grid_origin.y + (ijk.y + 0.5f) * _dx, _grid_origin.z + (ijk.z + 0.5f) * _dx };
        _psi_x[idx]    = pos;
        _T_x[idx]      = { 1.0f, 0.0f, 0.0f };
    }
}

__global__ void ResetToIdentityYKernel(float3* _psi_y, float3* _T_y, int3 _y_tile_dim, float3 _grid_origin, float _dx)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_y_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int idx        = tile_idx * 512 + voxel_idx;
        float3 pos     = { _grid_origin.x + (ijk.x + 0.5f) * _dx, _grid_origin.y + ijk.y * _dx, _grid_origin.z + (ijk.z + 0.5f) * _dx };
        _psi_y[idx]    = pos;
        _T_y[idx]      = { 0.0f, 1.0f, 0.0f };
    }
}

__global__ void ResetToIdentityZKernel(float3* _psi_z, float3* _T_z, int3 _z_tile_dim, float3 _grid_origin, float _dx)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_z_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int idx        = tile_idx * 512 + voxel_idx;
        float3 pos     = { _grid_origin.x + (ijk.x + 0.5f) * _dx, _grid_origin.y + (ijk.y + 0.5f) * _dx, _grid_origin.z + ijk.z * _dx };
        _psi_z[idx]    = pos;
        _T_z[idx]      = { 0.0f, 0.0f, 1.0f };
    }
}

void ResetToIdentityXASync(DHMemory<float3>& _psi_x, DHMemory<float3>& _T_x, int3 _x_tile_dim, float3 _grid_origin, float _dx, cudaStream_t _stream)
{
    float3* psi_x = _psi_x.dev_ptr_;
    float3* T_x   = _T_x.dev_ptr_;
    int tile_num  = Prod(_x_tile_dim);
    ResetToIdentityXKernel<<<tile_num, 128, 0, _stream>>>(psi_x, T_x, _x_tile_dim, _grid_origin, _dx);
}

void ResetToIdentityYASync(DHMemory<float3>& _psi_y, DHMemory<float3>& _T_y, int3 _y_tile_dim, float3 _grid_origin, float _dx, cudaStream_t _stream)
{
    float3* psi_y = _psi_y.dev_ptr_;
    float3* T_y   = _T_y.dev_ptr_;
    int tile_num  = Prod(_y_tile_dim);
    ResetToIdentityYKernel<<<tile_num, 128, 0, _stream>>>(psi_y, T_y, _y_tile_dim, _grid_origin, _dx);
}

void ResetToIdentityZASync(DHMemory<float3>& _psi_z, DHMemory<float3>& _T_z, int3 _z_tile_dim, float3 _grid_origin, float _dx, cudaStream_t _stream)
{
    float3* psi_z = _psi_z.dev_ptr_;
    float3* T_z   = _T_z.dev_ptr_;
    int tile_num  = Prod(_z_tile_dim);
    ResetToIdentityZKernel<<<tile_num, 128, 0, _stream>>>(psi_z, T_z, _z_tile_dim, _grid_origin, _dx);
}

__device__ __forceinline__ float N2(float x)
{
    float abs_x = fabsf(x);
    float ret   = abs_x < 0.5f ? (0.75f - abs_x * abs_x) : (0.5f * (1.5f - abs_x) * (1.5f - abs_x));
    return ret;
}

__device__ __forceinline__ float dN2(float x)
{
    float abs_x = fabsf(x);
    float ret   = abs_x < 0.5f ? (-2.0f * x) : (x - 1.5f * copysignf(1.0f, x));
    return ret;
}

__device__ float3 InterpMacN2Grad(float3x3& _grad, int3 _tile_dim, const float* _u_x, const float* _u_y, const float* _u_z, float3 _trans_pos, float _inv_dx)
{
    float eps               = 0.0001f;
    float3 ijk              = { _trans_pos.x * _inv_dx, _trans_pos.y * _inv_dx, _trans_pos.z * _inv_dx };
    float3 min_ijk          = { 0.5f + eps, 0.5f + eps, 0.5f + eps };
    float3 grid_dim         = { float(8 * _tile_dim.x), float(8 * _tile_dim.y), float(8 * _tile_dim.z) };
    float3 ret              = { 0.0f, 0.0f, 0.0f };
    // x
    int3 axis_tile_dim      = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    float3 axis_ijk         = { ijk.x, ijk.y - 0.5f, ijk.z - 0.5f };
    float3 clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 0.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 1.5f - eps });
    int3 base_ijk           = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    _grad.xx                = 0.0f;
    _grad.xy                = 0.0f;
    _grad.xz                = 0.0f;
    float N2_x[3], N2_y[3], N2_z[3];
    float dN2_x[3], dN2_y[3], dN2_z[3];
    int idx_x[3], idx_y[3], idx_z[3];

    int3 base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    int3 base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    int base_tile_idx  = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    int base_vol_idx   = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    int base_idx       = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        dN2_x[i]       = dN2(offset_x);
        dN2_y[i]       = dN2(offset_y);
        dN2_z[i]       = dN2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _u_x[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret.x += (val * N2_x_) * (N2_y_ * N2_z_);
                _grad.xx += val * dN2_x[i] * (N2_y_ * N2_z_);
                _grad.xy += (val * N2_x_) * dN2_y[j] * N2_z_;
                _grad.xz += (val * N2_x_) * N2_y_ * dN2_z[k];
            }
    _grad.xx *= _inv_dx;
    _grad.xy *= _inv_dx;
    _grad.xz *= _inv_dx;
    // y
    axis_tile_dim    = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    axis_ijk         = { ijk.x - 0.5f, ijk.y, ijk.z - 0.5f };
    clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 0.5f - eps, grid_dim.z - 1.5f - eps });
    base_ijk         = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    _grad.yx         = 0.0f;
    _grad.yy         = 0.0f;
    _grad.yz         = 0.0f;

    base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    base_tile_idx = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    base_vol_idx  = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    base_idx      = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        dN2_x[i]       = dN2(offset_x);
        dN2_y[i]       = dN2(offset_y);
        dN2_z[i]       = dN2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _u_y[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret.y += (val * N2_x_) * (N2_y_ * N2_z_);
                _grad.yx += val * dN2_x[i] * (N2_y_ * N2_z_);
                _grad.yy += (val * N2_x_) * dN2_y[j] * N2_z_;
                _grad.yz += (val * N2_x_) * N2_y_ * dN2_z[k];
            }
    _grad.yx *= _inv_dx;
    _grad.yy *= _inv_dx;
    _grad.yz *= _inv_dx;
    // z
    axis_tile_dim    = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    axis_ijk         = { ijk.x - 0.5f, ijk.y - 0.5f, ijk.z };
    clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 0.5f - eps });
    base_ijk         = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    _grad.zx         = 0.0f;
    _grad.zy         = 0.0f;
    _grad.zz         = 0.0f;

    base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    base_tile_idx = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    base_vol_idx  = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    base_idx      = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        dN2_x[i]       = dN2(offset_x);
        dN2_y[i]       = dN2(offset_y);
        dN2_z[i]       = dN2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _u_z[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret.z += (val * N2_x_) * (N2_y_ * N2_z_);
                _grad.zx += val * dN2_x[i] * (N2_y_ * N2_z_);
                _grad.zy += (val * N2_x_) * dN2_y[j] * N2_z_;
                _grad.zz += (val * N2_x_) * N2_y_ * dN2_z[k];
            }
    _grad.zx *= _inv_dx;
    _grad.zy *= _inv_dx;
    _grad.zz *= _inv_dx;
    return ret;
}

__global__ void RK2AxisKernel(float3* _psi_axis, float3* _T_axis, int3 _tile_dim, const float* _u_x,
                              const float* _u_y, const float* _u_z, float3 _grid_origin, float _inv_dx, float _dt)
{
    float half_dt = 0.5f * _dt;
    int tile_idx  = blockIdx.x;
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = t_id + i * 128;
        int idx       = tile_idx * 512 + voxel_idx;

        float3x3 grad;
        // first
        float3 pos        = _psi_axis[idx];
        float3 T1         = _T_axis[idx];
        float3 trans_pos  = { pos.x - _grid_origin.x, pos.y - _grid_origin.y, pos.z - _grid_origin.z };
        float3 u          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, trans_pos, _inv_dx);
        float3 dT_axis_dt = MatMulVec(grad, T1);
        // second
        trans_pos         = { pos.x - half_dt * u.x - _grid_origin.x, pos.y - half_dt * u.y - _grid_origin.y, pos.z - half_dt * u.z - _grid_origin.z };
        float3 T2         = { T1.x - half_dt * dT_axis_dt.x, T1.y - half_dt * dT_axis_dt.y, T1.z - half_dt * dT_axis_dt.z };
        u                 = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, trans_pos, _inv_dx);
        dT_axis_dt        = MatMulVec(grad, T2);
        //  final
        _psi_axis[idx]    = { pos.x - _dt * u.x, pos.y - _dt * u.y, pos.z - _dt * u.z };
        _T_axis[idx]      = { T1.x - _dt * dT_axis_dt.x, T1.y - _dt * dT_axis_dt.y, T1.z - _dt * dT_axis_dt.z };
    }
}

__global__ void RK4AxisKernel(float3* _psi_axis, float3* _T_axis, int3 _tile_dim, const float* _u_x,
                              const float* _u_y, const float* _u_z, float3 _grid_origin, float _inv_dx, float _dt)
{
    float half_dt         = 0.5f * _dt;
    float one_over_six_dt = 0.166667f * _dt;
    int tile_idx          = blockIdx.x;
    int t_id              = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = t_id + i * 128;
        int idx       = tile_idx * 512 + voxel_idx;

        float3x3 grad;
        // first
        float3 pos         = _psi_axis[idx];
        float3 T1          = _T_axis[idx];
        float3 trans_pos   = { pos.x - _grid_origin.x, pos.y - _grid_origin.y, pos.z - _grid_origin.z };
        float3 u1          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, trans_pos, _inv_dx);
        float3 dT_axis_dt1 = MatMulVec(grad, T1);
        // second
        float3 intp_pos    = { trans_pos.x - half_dt * u1.x, trans_pos.y - half_dt * u1.y, trans_pos.z - half_dt * u1.z };
        float3 T           = { T1.x - half_dt * dT_axis_dt1.x, T1.y - half_dt * dT_axis_dt1.y, T1.z - half_dt * dT_axis_dt1.z };
        float3 u2          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, intp_pos, _inv_dx);
        float3 dT_axis_dt2 = MatMulVec(grad, T);
        // third
        intp_pos           = { trans_pos.x - half_dt * u2.x, trans_pos.y - half_dt * u2.y, trans_pos.z - half_dt * u2.z };
        T                  = { T1.x - half_dt * dT_axis_dt2.x, T1.y - half_dt * dT_axis_dt2.y, T1.z - half_dt * dT_axis_dt2.z };
        float3 u3          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, intp_pos, _inv_dx);
        float3 dT_axis_dt3 = MatMulVec(grad, T);
        //  forth
        intp_pos           = { trans_pos.x - _dt * u3.x, trans_pos.y - _dt * u3.y, trans_pos.z - _dt * u3.z };
        T                  = { T1.x - _dt * dT_axis_dt3.x, T1.y - _dt * dT_axis_dt3.y, T1.z - _dt * dT_axis_dt3.z };
        float3 u4          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, intp_pos, _inv_dx);
        float3 dT_axis_dt4 = MatMulVec(grad, T);
        // final
        _psi_axis[idx]     = { trans_pos.x - one_over_six_dt * (u1.x + 2.0f * u2.x + 2.0f * u3.x + u4.x),
                               pos.y - one_over_six_dt * (u1.y + 2.0f * u2.y + 2.0f * u3.y + u4.y),
                               pos.z - one_over_six_dt * (u1.z + 2.0f * u2.z + 2.0f * u3.z + u4.z) };
        _T_axis[idx]       = { T1.x - one_over_six_dt * (dT_axis_dt1.x + 2.0f * dT_axis_dt2.x + 2.0f * dT_axis_dt3.x + dT_axis_dt4.x),
                               T1.y - one_over_six_dt * (dT_axis_dt1.y + 2.0f * dT_axis_dt2.y + 2.0f * dT_axis_dt3.y + dT_axis_dt4.y),
                               T1.z - one_over_six_dt * (dT_axis_dt1.z + 2.0f * dT_axis_dt2.z + 2.0f * dT_axis_dt3.z + dT_axis_dt4.z) };
    }
}

void RKAxisAsync(int _rk_order, DHMemory<float3>& _psi_axis, DHMemory<float3>& _T_axis, int3 _tile_dim, int3 _axis_tile_dim,
                 const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, float3 _grid_origin, float _dx, float _dt, cudaStream_t _stream)
{
    float3* psi_axis  = _psi_axis.dev_ptr_;
    float3* T_axis    = _T_axis.dev_ptr_;
    const float* u_x  = _u_x.dev_ptr_;
    const float* u_y  = _u_y.dev_ptr_;
    const float* u_z  = _u_z.dev_ptr_;
    int axis_tile_num = Prod(_axis_tile_dim);
    float inv_dx      = 1.0f / _dx;
    if (_rk_order == 2)
        RK2AxisKernel<<<axis_tile_num, 128, 0, _stream>>>(psi_axis, T_axis, _tile_dim, u_x, u_y, u_z, _grid_origin, inv_dx, _dt);
    else if (_rk_order == 4)
        RK4AxisKernel<<<axis_tile_num, 128, 0, _stream>>>(psi_axis, T_axis, _tile_dim, u_x, u_y, u_z, _grid_origin, inv_dx, _dt);
}

__device__ float3 InterpMacN2(int3 _tile_dim, const float* _u_x, const float* _u_y, const float* _u_z, float3 _trans_pos, float _inv_dx)
{
    float eps               = 0.0001f;
    float3 ijk              = { _trans_pos.x * _inv_dx, _trans_pos.y * _inv_dx, _trans_pos.z * _inv_dx };
    float3 min_ijk          = { 0.5f + eps, 0.5f + eps, 0.5f + eps };
    float3 grid_dim         = { float(8 * _tile_dim.x), float(8 * _tile_dim.y), float(8 * _tile_dim.z) };
    float3 ret              = { 0.0f, 0.0f, 0.0f };
    // x
    int3 axis_tile_dim      = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    float3 axis_ijk         = { ijk.x, ijk.y - 0.5f, ijk.z - 0.5f };
    float3 clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 0.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 1.5f - eps });
    int3 base_ijk           = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    float N2_x[3], N2_y[3], N2_z[3];
    int idx_x[3], idx_y[3], idx_z[3];

    int3 base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    int3 base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    int base_tile_idx  = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    int base_vol_idx   = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    int base_idx       = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _u_x[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret.x += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    // y
    axis_tile_dim    = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    axis_ijk         = { ijk.x - 0.5f, ijk.y, ijk.z - 0.5f };
    clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 0.5f - eps, grid_dim.z - 1.5f - eps });
    base_ijk         = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };

    base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    base_tile_idx = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    base_vol_idx  = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    base_idx      = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _u_y[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret.y += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    // z
    axis_tile_dim    = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    axis_ijk         = { ijk.x - 0.5f, ijk.y - 0.5f, ijk.z };
    clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 0.5f - eps });
    base_ijk         = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };

    base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    base_tile_idx = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    base_vol_idx  = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    base_idx      = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _u_z[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret.z += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    return ret;
}

__global__ void PullbackAxisKernel(float* _dst_axis, int3 _tile_dim, const float* _src_x, const float* _src_y, const float* _src_z, const float3* _psi_axis, const float3* T_axis, float3 _grid_origin, float _inv_dx)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int idx        = tile_idx * 512 + voxel_idx;
        float3 pos     = _psi_axis[idx];
        pos            = { pos.x - _grid_origin.x, pos.y - _grid_origin.y, pos.z - _grid_origin.z };
        float3 src     = InterpMacN2(_tile_dim, _src_x, _src_y, _src_z, pos, _inv_dx);
        float3 T       = T_axis[idx];
        _dst_axis[idx] = T.x * src.x + T.y * src.y + T.z * src.z;
    }
}

void PullbackAxisAsync(DHMemory<float>& _dst_axis, int3 _tile_dim, int3 _axis_tile_dim, const DHMemory<float>& _src_x, const DHMemory<float>& _src_y, const DHMemory<float>& _src_z,
                       const DHMemory<float3>& _psi_axis, const DHMemory<float3>& _T_axis, float3 _grid_origin, float _dx, cudaStream_t _stream)
{
    float* dst_axis        = _dst_axis.dev_ptr_;
    const float* src_x     = _src_x.dev_ptr_;
    const float* src_y     = _src_y.dev_ptr_;
    const float* src_z     = _src_z.dev_ptr_;
    const float3* psi_axis = _psi_axis.dev_ptr_;
    const float3* T_axis   = _T_axis.dev_ptr_;
    int axis_tile_num      = Prod(_axis_tile_dim);
    float inv_dx           = 1.0f / _dx;
    PullbackAxisKernel<<<axis_tile_num, 128, 0, _stream>>>(dst_axis, _tile_dim, src_x, src_y, src_z, psi_axis, T_axis, _grid_origin, inv_dx);
}

__global__ void PullbackCenterKernel(float* _dst, int3 _tile_dim, const float* _src, const float3* _psi_c, float3 _grid_origin, float _inv_dx)
{
    int tile_idx = blockIdx.x;
    int3 max_ijk = { 8 * _tile_dim.x - 1, 8 * _tile_dim.y - 1, 8 * _tile_dim.z - 1 };
    int t_id     = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = t_id + i * 128;
        int idx       = tile_idx * 512 + voxel_idx;
        float3 pos    = _psi_c[idx];
        pos           = { pos.x - _grid_origin.x, pos.y - _grid_origin.y, pos.z - _grid_origin.z };
        float3 ijk    = { pos.x * _inv_dx - 0.5f, pos.y * _inv_dx - 0.5f, pos.z * _inv_dx - 0.5f };
        _dst[idx]     = Interp(ijk, _src, _tile_dim, max_ijk);
    }
}

void PullbackCenterAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float3>& _psi_c, float3 _grid_origin, float _dx, cudaStream_t _stream)
{
    float* dst          = _dst.dev_ptr_;
    const float* src    = _src.dev_ptr_;
    const float3* psi_c = _psi_c.dev_ptr_;
    int tile_num        = Prod(_tile_dim);
    float inv_dx        = 1.0f / _dx;
    PullbackCenterKernel<<<tile_num, 128, 0, _stream>>>(dst, _tile_dim, src, psi_c, _grid_origin, inv_dx);
}

__global__ void GetCentralPsiKernel(float3* _psi_c, int3 _tile_dim, const float3* _psi_x, const float3* _psi_y, const float3* _psi_z)
{
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id        = threadIdx.x;
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    for (int i = 0; i < 4; i++) {
        int voxel_idx     = t_id + i * 128;
        int3 voxel_ijk    = VoxelIdxToIjk(voxel_idx);
        int idx           = tile_idx * 512 + voxel_idx;
        int3 ijk          = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float3 ans        = { 0.0f, 0.0f, 0.0f };
        int target_idx    = IjkToIdx(x_tile_dim, ijk);
        float3 target_psi = _psi_x[target_idx];
        ans               = target_psi;
        target_idx        = IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z });
        target_psi        = _psi_x[target_idx];
        ans               = { ans.x + target_psi.x, ans.y + target_psi.y, ans.z + target_psi.z };
        target_idx        = IjkToIdx(y_tile_dim, ijk);
        target_psi        = _psi_y[target_idx];
        ans               = { ans.x + target_psi.x, ans.y + target_psi.y, ans.z + target_psi.z };
        target_idx        = IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z });
        target_psi        = _psi_y[target_idx];
        ans               = { ans.x + target_psi.x, ans.y + target_psi.y, ans.z + target_psi.z };
        target_idx        = IjkToIdx(z_tile_dim, ijk);
        target_psi        = _psi_z[target_idx];
        ans               = { ans.x + target_psi.x, ans.y + target_psi.y, ans.z + target_psi.z };
        target_idx        = IjkToIdx(z_tile_dim, { ijk.x, ijk.y, ijk.z + 1 });
        target_psi        = _psi_z[target_idx];
        ans               = { ans.x + target_psi.x, ans.y + target_psi.y, ans.z + target_psi.z };
        _psi_c[idx]       = { ans.x * 0.166667f, ans.y * 0.166667f, ans.z * 0.166667f };
    }
}

void GetCentralPsiAsync(DHMemory<float3>& _psi_c, int3 _tile_dim, const DHMemory<float3>& _psi_x, const DHMemory<float3>& _psi_y, const DHMemory<float3>& _psi_z, cudaStream_t _stream)
{
    float3* psi_c       = _psi_c.dev_ptr_;
    const float3* psi_x = _psi_x.dev_ptr_;
    const float3* psi_y = _psi_y.dev_ptr_;
    const float3* psi_z = _psi_z.dev_ptr_;
    int tile_num        = Prod(_tile_dim);
    GetCentralPsiKernel<<<tile_num, 128, 0, _stream>>>(psi_c, _tile_dim, psi_x, psi_y, psi_z);
}

__global__ void BfeccClampKernel(float* _after_bfecc, int3 _tile_dim, int3 _max_ijk, const float* _before_bfecc)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx   = t_id + i * 128;
        int3 voxel_ijk  = VoxelIdxToIjk(voxel_idx);
        int idx         = tile_idx * 512 + voxel_idx;
        float after_val = _after_bfecc[idx];
        int3 ijk        = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int3 start_ijk  = { (ijk.x - 1 < 0) ? 0 : (ijk.x - 1), (ijk.y - 1) < 0 ? 0 : (ijk.y - 1), (ijk.z - 1) < 0 ? 0 : (ijk.z - 1) };
         int3 end_ijk = { (ijk.x + 1 > _max_ijk.x) ? _max_ijk.x : (ijk.x + 1), (ijk.y + 1 > _max_ijk.y) ? _max_ijk.y : (ijk.y + 1), (ijk.z + 1 > _max_ijk.z) ? _max_ijk.z : (ijk.z + 1) };
         float min_val, max_val;
        bool first = true;
        for (int a = start_ijk.x; a <= end_ijk.x; a++)
            for (int b = start_ijk.y; b <= end_ijk.y; b++)
                for (int c = start_ijk.z; c <= end_ijk.z; c++)
                    if (abs(a - ijk.x) + abs(b - ijk.y) + abs(c - ijk.z) == 1) {
                        int target_idx   = IjkToIdx(_tile_dim, { a, b, c });
                        float before_val = _before_bfecc[target_idx];
                        if (first) {
                            min_val = before_val;
                            max_val = before_val;
                            first   = false;
                        } else {
                            min_val = (min_val < before_val) ? min_val : before_val;
                            max_val = (max_val > before_val) ? max_val : before_val;
                        }
                    }
        _after_bfecc[idx] = Clamp(after_val, min_val, max_val);
    }
}

void BfeccClampAsync(DHMemory<float>& _after_bfecc, int3 _tile_dim, int3 _max_ijk, const DHMemory<float>& _before_bfecc, cudaStream_t _stream)
{
    float* after_bfecc        = _after_bfecc.dev_ptr_;
    const float* before_bfecc = _before_bfecc.dev_ptr_;
    int tile_num              = Prod(_tile_dim);
    BfeccClampKernel<<<tile_num, 128, 0, _stream>>>(after_bfecc, _tile_dim, _max_ijk, before_bfecc);
}

__global__ void RK2AxisAccumulateForceKernel(float3* _psi_axis, float3* _T_axis, float* _f_axis, int3 _tile_dim, const float* _u_x,
                                             const float* _u_y, const float* _u_z, const float* _f_x, const float* _f_y, const float* _f_z, float3 _grid_origin, float _inv_dx, float _dt)
{
    float half_dt = 0.5f * _dt;
    int tile_idx  = blockIdx.x;
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = t_id + i * 128;
        int idx       = tile_idx * 512 + voxel_idx;

        float3x3 grad;
        // first
        float3 pos        = _psi_axis[idx];
        float3 T1         = _T_axis[idx];
        float3 trans_pos  = { pos.x - _grid_origin.x, pos.y - _grid_origin.y, pos.z - _grid_origin.z };
        float3 u          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, trans_pos, _inv_dx);
        float3 dT_axis_dt = MatMulVec(grad, T1);

        float3 f       = InterpMacN2(_tile_dim, _f_x, _f_y, _f_z, trans_pos, _inv_dx);
        _f_axis[idx]   = f.x * T1.x + f.y * T1.y + f.z * T1.z;
        // second
        trans_pos      = { pos.x - half_dt * u.x - _grid_origin.x, pos.y - half_dt * u.y - _grid_origin.y, pos.z - half_dt * u.z - _grid_origin.z };
        float3 T2      = { T1.x - half_dt * dT_axis_dt.x, T1.y - half_dt * dT_axis_dt.y, T1.z - half_dt * dT_axis_dt.z };
        u              = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, trans_pos, _inv_dx);
        dT_axis_dt     = MatMulVec(grad, T2);
        //  final
        _psi_axis[idx] = { pos.x - _dt * u.x, pos.y - _dt * u.y, pos.z - _dt * u.z };
        _T_axis[idx]   = { T1.x - _dt * dT_axis_dt.x, T1.y - _dt * dT_axis_dt.y, T1.z - _dt * dT_axis_dt.z };
    }
}

__global__ void RK4AxisAccumulateForceKernel(float3* _psi_axis, float3* _T_axis, float* _f_axis, int3 _tile_dim, const float* _u_x,
                                             const float* _u_y, const float* _u_z, const float* _f_x, const float* _f_y, const float* _f_z, float3 _grid_origin, float _inv_dx, float _dt)
{
    float half_dt         = 0.5f * _dt;
    float one_over_six_dt = 0.166667f * _dt;
    int tile_idx          = blockIdx.x;
    int t_id              = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = t_id + i * 128;
        int idx       = tile_idx * 512 + voxel_idx;

        float3x3 grad;
        // first
        float3 pos         = _psi_axis[idx];
        float3 T1          = _T_axis[idx];
        float3 trans_pos   = { pos.x - _grid_origin.x, pos.y - _grid_origin.y, pos.z - _grid_origin.z };
        float3 u1          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, trans_pos, _inv_dx);
        float3 dT_axis_dt1 = MatMulVec(grad, T1);

        float3 f           = InterpMacN2(_tile_dim, _f_x, _f_y, _f_z, trans_pos, _inv_dx);
        _f_axis[idx]       = f.x * T1.x + f.y * T1.y + f.z * T1.z;
        // second
        float3 intp_pos    = { trans_pos.x - half_dt * u1.x, trans_pos.y - half_dt * u1.y, trans_pos.z - half_dt * u1.z };
        float3 T           = { T1.x - half_dt * dT_axis_dt1.x, T1.y - half_dt * dT_axis_dt1.y, T1.z - half_dt * dT_axis_dt1.z };
        float3 u2          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, intp_pos, _inv_dx);
        float3 dT_axis_dt2 = MatMulVec(grad, T);
        // third
        intp_pos           = { trans_pos.x - half_dt * u2.x, trans_pos.y - half_dt * u2.y, trans_pos.z - half_dt * u2.z };
        T                  = { T1.x - half_dt * dT_axis_dt2.x, T1.y - half_dt * dT_axis_dt2.y, T1.z - half_dt * dT_axis_dt2.z };
        float3 u3          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, intp_pos, _inv_dx);
        float3 dT_axis_dt3 = MatMulVec(grad, T);
        //  forth
        intp_pos           = { trans_pos.x - _dt * u3.x, trans_pos.y - _dt * u3.y, trans_pos.z - _dt * u3.z };
        T                  = { T1.x - _dt * dT_axis_dt3.x, T1.y - _dt * dT_axis_dt3.y, T1.z - _dt * dT_axis_dt3.z };
        float3 u4          = InterpMacN2Grad(grad, _tile_dim, _u_x, _u_y, _u_z, intp_pos, _inv_dx);
        float3 dT_axis_dt4 = MatMulVec(grad, T);
        // final
        _psi_axis[idx]     = { trans_pos.x - one_over_six_dt * (u1.x + 2.0f * u2.x + 2.0f * u3.x + u4.x),
                               pos.y - one_over_six_dt * (u1.y + 2.0f * u2.y + 2.0f * u3.y + u4.y),
                               pos.z - one_over_six_dt * (u1.z + 2.0f * u2.z + 2.0f * u3.z + u4.z) };
        _T_axis[idx]       = { T1.x - one_over_six_dt * (dT_axis_dt1.x + 2.0f * dT_axis_dt2.x + 2.0f * dT_axis_dt3.x + dT_axis_dt4.x),
                               T1.y - one_over_six_dt * (dT_axis_dt1.y + 2.0f * dT_axis_dt2.y + 2.0f * dT_axis_dt3.y + dT_axis_dt4.y),
                               T1.z - one_over_six_dt * (dT_axis_dt1.z + 2.0f * dT_axis_dt2.z + 2.0f * dT_axis_dt3.z + dT_axis_dt4.z) };
    }
}

void RKAxisAccumulateForceAsync(int _rk_order, DHMemory<float3>& _psi_axis, DHMemory<float3>& _T_axis, DHMemory<float>& _f_axis, int3 _tile_dim, int3 _axis_tile_dim,
                                const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, const DHMemory<float>& _f_x, const DHMemory<float>& _f_y, const DHMemory<float>& _f_z, float3 _grid_origin, float _dx, float _dt, cudaStream_t _stream)
{
    float3* psi_axis  = _psi_axis.dev_ptr_;
    float3* T_axis    = _T_axis.dev_ptr_;
    float* f_axis     = _f_axis.dev_ptr_;
    const float* u_x  = _u_x.dev_ptr_;
    const float* u_y  = _u_y.dev_ptr_;
    const float* u_z  = _u_z.dev_ptr_;
    const float* f_x  = _f_x.dev_ptr_;
    const float* f_y  = _f_y.dev_ptr_;
    const float* f_z  = _f_z.dev_ptr_;
    int axis_tile_num = Prod(_axis_tile_dim);
    float inv_dx      = 1.0f / _dx;
    if (_rk_order == 2)
        RK2AxisAccumulateForceKernel<<<axis_tile_num, 128, 0, _stream>>>(psi_axis, T_axis, f_axis, _tile_dim, u_x, u_y, u_z, f_x, f_y, f_z, _grid_origin, inv_dx, _dt);
    else if (_rk_order == 4)
        RK4AxisAccumulateForceKernel<<<axis_tile_num, 128, 0, _stream>>>(psi_axis, T_axis, f_axis, _tile_dim, u_x, u_y, u_z, f_x, f_y, f_z, _grid_origin, inv_dx, _dt);
}

__global__ void AddFieldsKernel(float* _dst, float* _src1, float* _src2, float _coef2)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        int idx       = tile_idx * 512 + voxel_idx;
        _dst[idx]     = _src1[idx] + _coef2 * _src2[idx];
    }
}

void AddFieldsAsync(DHMemory<float>& _dst, int3 _tile_dim, DHMemory<float>& _src1, DHMemory<float>& _src2, float _coef2, cudaStream_t _stream)
{
    int tile_num = Prod(_tile_dim);
    AddFieldsKernel<<<tile_num, 128, 0, _stream>>>(_dst.dev_ptr_, _src1.dev_ptr_, _src2.dev_ptr_, _coef2);
}

__global__ void SetBcAxisKernel(float* _u_axis, const uint8_t* _is_bc_axis, const float* _bc_val_axis)
{
    int tile_idx  = blockIdx.x;
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        int idx       = tile_idx * 512 + voxel_idx;
        if (_is_bc_axis[idx]) {
            _u_axis[idx] = _bc_val_axis[idx];
        }
    }
}

void SetBcAxisAsync(DHMemory<float>& _u_axis, int3 _axis_tile_dim, const DHMemory<uint8_t>& _is_bc_axis, const DHMemory<float>& _bc_val_axis, cudaStream_t _stream)
{
    float* u_axis            = _u_axis.dev_ptr_;
    const uint8_t* is_bc_axis   = _is_bc_axis.dev_ptr_;
    const float* bc_val_axis = _bc_val_axis.dev_ptr_;
    int axis_tile_num        = Prod(_axis_tile_dim);
    SetBcAxisKernel<<<axis_tile_num, 128, 0, _stream>>>(u_axis, is_bc_axis, bc_val_axis);
}

__global__ void CalcDivKernel(float* _b, int3 _tile_dim, const uint8_t* _is_dof, const float* _u_x, const float* _u_y, const float* _u_z)
{
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(_tile_dim, tile_idx);
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        int idx       = tile_idx * 512 + voxel_idx;
        if (_is_dof[idx]) {
            int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
            int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
            int3 x_ijk     = { ijk.x + 1, ijk.y, ijk.z };
            int3 y_ijk     = { ijk.x, ijk.y + 1, ijk.z };
            int3 z_ijk     = { ijk.x, ijk.y, ijk.z + 1 };
            float b        = 0.0f;
            b += _u_x[IjkToIdx(x_tile_dim, ijk)];
            b -= _u_x[IjkToIdx(x_tile_dim, x_ijk)];
            b += _u_y[IjkToIdx(y_tile_dim, ijk)];
            b -= _u_y[IjkToIdx(y_tile_dim, y_ijk)];
            b += _u_z[IjkToIdx(z_tile_dim, ijk)];
            b -= _u_z[IjkToIdx(z_tile_dim, z_ijk)];
            _b[idx] = b;
        } else
            _b[idx] = 0.0f;
    }
}

void CalcDivAsync(DHMemory<float>& _b, int3 _tile_dim, const DHMemory<uint8_t>& _is_dof, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z, cudaStream_t _stream)
{
    float* b           = _b.dev_ptr_;
    const uint8_t* is_dof = _is_dof.dev_ptr_;
    const float* u_x   = _u_x.dev_ptr_;
    const float* u_y   = _u_y.dev_ptr_;
    const float* u_z   = _u_z.dev_ptr_;
    int tile_num       = Prod(_tile_dim);
    CalcDivKernel<<<tile_num, 128, 0, _stream>>>(b, _tile_dim, is_dof, u_x, u_y, u_z);
}

__global__ void ApplyPressureXKernel(float* _u_x, int3 _tile_dim, const float* _p, const uint8_t* _is_bc_x)
{
    int b_id        = blockIdx.x;
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 tile_ijk   = TileIdxToIjk(x_tile_dim, b_id);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int x_idx      = IjkToIdx(x_tile_dim, ijk);
        float left_p   = 0.0f;
        float right_p  = 0.0f;
        if (!_is_bc_x[x_idx]) {
            if (ijk.x > 0) {
                int3 left_ijk = { ijk.x - 1, ijk.y, ijk.z };
                int left_idx  = IjkToIdx(_tile_dim, left_ijk);
                left_p        = _p[left_idx];
            }
            if (ijk.x < _tile_dim.x * 8) {
                int3 right_ijk = { ijk.x, ijk.y, ijk.z };
                int right_idx  = IjkToIdx(_tile_dim, right_ijk);
                right_p        = _p[right_idx];
            }
            _u_x[x_idx] += (left_p - right_p);
        }
    }
}

__global__ void ApplyPressureYKernel(float* _u_y, int3 _tile_dim, const float* _p, const uint8_t* _is_bc_y)
{
    int b_id        = blockIdx.x;
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 tile_ijk   = TileIdxToIjk(y_tile_dim, b_id);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int y_idx      = IjkToIdx(y_tile_dim, ijk);
        float bottom_p = 0.0f;
        float top_p    = 0.0f;
        if (!_is_bc_y[y_idx]) {
            if (ijk.y > 0) {
                int3 bottom_ijk = { ijk.x, ijk.y - 1, ijk.z };
                int bottom_idx  = IjkToIdx(_tile_dim, bottom_ijk);
                bottom_p        = _p[bottom_idx];
            }
            if (ijk.y < _tile_dim.y * 8) {
                int3 top_ijk = { ijk.x, ijk.y, ijk.z };
                int top_idx  = IjkToIdx(_tile_dim, top_ijk);
                top_p        = _p[top_idx];
            }
            _u_y[y_idx] += (bottom_p - top_p);
        }
    }
}

__global__ void ApplyPressureZKernel(float* _u_z, int3 _tile_dim, const float* _p, const uint8_t* _is_bc_z)
{
    int b_id        = blockIdx.x;
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    int3 tile_ijk   = TileIdxToIjk(z_tile_dim, b_id);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int z_idx      = IjkToIdx(z_tile_dim, ijk);
        float back_p   = 0.0f;
        float front_p  = 0.0f;
        if (!_is_bc_z[z_idx]) {
            if (ijk.z > 0) {
                int3 back_ijk = { ijk.x, ijk.y, ijk.z - 1 };
                int back_idx  = IjkToIdx(_tile_dim, back_ijk);
                back_p        = _p[back_idx];
            }
            if (ijk.z < _tile_dim.z * 8) {
                int3 front_ijk = { ijk.x, ijk.y, ijk.z };
                int front_idx  = IjkToIdx(_tile_dim, front_ijk);
                front_p        = _p[front_idx];
            }
            _u_z[z_idx] += (back_p - front_p);
        }
    }
}

void ApplyPressureAsync(DHMemory<float>& _u_x, DHMemory<float>& _u_y, DHMemory<float>& _u_z, int3 _tile_dim, const DHMemory<float>& _p, const DHMemory<uint8_t>& _is_bc_x, const DHMemory<uint8_t>& _is_bc_y, const DHMemory<uint8_t>& _is_bc_z, cudaStream_t _stream)
{
    int3 x_tile_dim     = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim     = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim     = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    int x_tile_num      = Prod(x_tile_dim);
    int y_tile_num      = Prod(y_tile_dim);
    int z_tile_num      = Prod(z_tile_dim);
    float* u_x          = _u_x.dev_ptr_;
    float* u_y          = _u_y.dev_ptr_;
    float* u_z          = _u_z.dev_ptr_;
    const float* p      = _p.dev_ptr_;
    const uint8_t* is_bc_x = _is_bc_x.dev_ptr_;
    const uint8_t* is_bc_y = _is_bc_y.dev_ptr_;
    const uint8_t* is_bc_z = _is_bc_z.dev_ptr_;
    ApplyPressureXKernel<<<x_tile_num, 128, 0, _stream>>>(u_x, _tile_dim, p, is_bc_x);
    ApplyPressureYKernel<<<y_tile_num, 128, 0, _stream>>>(u_y, _tile_dim, p, is_bc_y);
    ApplyPressureZKernel<<<z_tile_num, 128, 0, _stream>>>(u_z, _tile_dim, p, is_bc_z);
}

__global__ void SetWallBcXKernel(uint8_t* _is_bc_x, float* _bc_val_x, int3 _x_tile_dim, float _neg_bc_val_x, float _pos_bc_val_x)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_x_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    int3 grid_dim = { (_x_tile_dim.x - 1) * 8, _x_tile_dim.y * 8, _x_tile_dim.z * 8 };
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int idx        = tile_idx * 512 + voxel_idx;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.x == 0) {
            _is_bc_x[idx]  = 1;
            _bc_val_x[idx] = _neg_bc_val_x;
        } else if (ijk.x >= grid_dim.x) {
            _is_bc_x[idx]  = 1;
            _bc_val_x[idx] = _pos_bc_val_x;
        } else
            _is_bc_x[idx] = 0;
    }
}

__global__ void SetWallBcYKernel(uint8_t* _is_bc_y, float* _bc_val_y, int3 _y_tile_dim, float _neg_bc_val_y, float _pos_bc_val_y)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_y_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    int3 grid_dim = { _y_tile_dim.x * 8, (_y_tile_dim.y - 1) * 8, _y_tile_dim.z * 8 };
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int idx        = tile_idx * 512 + voxel_idx;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.y == 0) {
            _is_bc_y[idx]  = 1;
            _bc_val_y[idx] = _neg_bc_val_y;
        } else if (ijk.y >= grid_dim.y) {
            _is_bc_y[idx]  = 1;
            _bc_val_y[idx] = _pos_bc_val_y;
        } else
            _is_bc_y[idx] = 0;
    }
}

__global__ void SetWallBcZKernel(uint8_t* _is_bc_z, float* _bc_val_z, int3 _z_tile_dim, float _neg_bc_val_z, float _pos_bc_val_z)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_z_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    int3 grid_dim = { _z_tile_dim.x * 8, _z_tile_dim.y * 8, (_z_tile_dim.z - 1) * 8 };
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int idx        = tile_idx * 512 + voxel_idx;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (ijk.z == 0) {
            _is_bc_z[idx]  = 1;
            _bc_val_z[idx] = _neg_bc_val_z;
        } else if (ijk.z >= grid_dim.z) {
            _is_bc_z[idx]  = 1;
            _bc_val_z[idx] = _pos_bc_val_z;
        } else
            _is_bc_z[idx] = 0;
    }
}

void SetWallBcAsync(DHMemory<uint8_t>& _is_bc_x, DHMemory<uint8_t>& _is_bc_y, DHMemory<uint8_t>& _is_bc_z, DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, DHMemory<float>& _bc_val_z, int3 _tile_dim,
                    float3 _neg_bc_val, float3 _pos_bc_val, cudaStream_t _stream)
{
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    uint8_t* is_bc_x   = _is_bc_x.dev_ptr_;
    uint8_t* is_bc_y   = _is_bc_y.dev_ptr_;
    uint8_t* is_bc_z   = _is_bc_z.dev_ptr_;
    float* bc_val_x = _bc_val_x.dev_ptr_;
    float* bc_val_y = _bc_val_y.dev_ptr_;
    float* bc_val_z = _bc_val_z.dev_ptr_;
    SetWallBcXKernel<<<Prod(x_tile_dim), 128, 0, _stream>>>(is_bc_x, bc_val_x, x_tile_dim, _neg_bc_val.x, _pos_bc_val.x);
    SetWallBcYKernel<<<Prod(y_tile_dim), 128, 0, _stream>>>(is_bc_y, bc_val_y, y_tile_dim, _neg_bc_val.y, _pos_bc_val.y);
    SetWallBcZKernel<<<Prod(z_tile_dim), 128, 0, _stream>>>(is_bc_z, bc_val_z, z_tile_dim, _neg_bc_val.z, _pos_bc_val.z);
}

__global__ void SetBcByPhiKernel(uint8_t* _is_bc_x, uint8_t* _is_bc_y, uint8_t* _is_bc_z, float* _bc_val_x, float* _bc_val_y, float* _bc_val_z, int3 _tile_dim, const float* _phi)
{
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(_tile_dim, tile_idx);
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int idx        = tile_idx * 512 + voxel_idx;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        if (_phi[idx] < 0.0f) {
            _is_bc_x[IjkToIdx(x_tile_dim, ijk)]                          = 1;
            _is_bc_y[IjkToIdx(y_tile_dim, ijk)]                          = 1;
            _is_bc_z[IjkToIdx(z_tile_dim, ijk)]                          = 1;
            _is_bc_x[IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z })]  = 1;
            _is_bc_y[IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z })]  = 1;
            _is_bc_z[IjkToIdx(z_tile_dim, { ijk.x, ijk.y, ijk.z + 1 })]  = 1;
            _bc_val_x[IjkToIdx(x_tile_dim, ijk)]                         = 0.0f;
            _bc_val_y[IjkToIdx(y_tile_dim, ijk)]                         = 0.0f;
            _bc_val_z[IjkToIdx(z_tile_dim, ijk)]                         = 0.0f;
            _bc_val_x[IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z })] = 0.0f;
            _bc_val_y[IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z })] = 0.0f;
            _bc_val_z[IjkToIdx(z_tile_dim, { ijk.x, ijk.y, ijk.z + 1 })] = 0.0f;
        }
    }
}

void SetBcByPhiAsync(DHMemory<uint8_t>& _is_bc_x, DHMemory<uint8_t>& _is_bc_y, DHMemory<uint8_t>& _is_bc_z, DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, DHMemory<float>& _bc_val_z, int3 _tile_dim, const DHMemory<float>& _phi, cudaStream_t _stream)
{
    uint8_t* is_bc_x    = _is_bc_x.dev_ptr_;
    uint8_t* is_bc_y    = _is_bc_y.dev_ptr_;
    uint8_t* is_bc_z    = _is_bc_z.dev_ptr_;
    float* bc_val_x  = _bc_val_x.dev_ptr_;
    float* bc_val_y  = _bc_val_y.dev_ptr_;
    float* bc_val_z  = _bc_val_z.dev_ptr_;
    const float* phi = _phi.dev_ptr_;
    SetBcByPhiKernel<<<Prod(_tile_dim), 128, 0, _stream>>>(is_bc_x, is_bc_y, is_bc_z, bc_val_x, bc_val_y, bc_val_z, _tile_dim, phi);
}

__global__ void SetBcBySurfaceKernel(uint8_t* _is_bc_x, uint8_t* _is_bc_y, uint8_t* _is_bc_z, float* _bc_val_x, float* _bc_val_y, float* _bc_val_z, int3 _tile_dim, cudaSurfaceObject_t voxel_surface, cudaSurfaceObject_t velocity_surface)
{
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(_tile_dim, tile_idx);
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; ++i) {
        int voxel_idx    = t_id + i * 128;
        int3 voxel_ijk   = VoxelIdxToIjk(voxel_idx);
        int3 ijk         = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };

        // Note the order of coordinates: x, z, y, since each slice is a x-z plane
        auto boundary = surf3Dread<uint8_t>(voxel_surface, ijk.x, ijk.z, ijk.y, cudaBoundaryModeTrap);
        if (boundary != 0) {
            auto boundary_velocity = surf3Dread<float4>(velocity_surface, ijk.x * 16, ijk.z, ijk.y, cudaBoundaryModeTrap);
            //printf("boundary velocity: (%f, %f, %f) at (%d, %d, %d)\n", boundary_velocity.x, boundary_velocity.y, boundary_velocity.z, ijk.x, ijk.y, ijk.z);
            _is_bc_x[IjkToIdx(x_tile_dim, ijk)]                                  = 1;
            _is_bc_y[IjkToIdx(y_tile_dim, ijk)]                                  = 1;
            _is_bc_z[IjkToIdx(z_tile_dim, ijk)]                                  = 1;
            _is_bc_x[IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z })] = 1;
            _is_bc_y[IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z })] = 1;
            _is_bc_z[IjkToIdx(z_tile_dim, { ijk.x, ijk.y, ijk.z + 1 })] = 1;
            _bc_val_x[IjkToIdx(x_tile_dim, ijk)]                                  = boundary_velocity.x;
            _bc_val_y[IjkToIdx(y_tile_dim, ijk)]                                  = boundary_velocity.y;
            _bc_val_z[IjkToIdx(z_tile_dim, ijk)]                                  = boundary_velocity.z;
            _bc_val_x[IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z })] = boundary_velocity.x;
            _bc_val_y[IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z })] = boundary_velocity.y;
            _bc_val_z[IjkToIdx(z_tile_dim, { ijk.x, ijk.y, ijk.z + 1 })] = boundary_velocity.z;
        }
    }
}

void SetBcBySurfaceAsync(DHMemory<uint8_t>& _is_bc_x, DHMemory<uint8_t>& _is_bc_y, DHMemory<uint8_t>& _is_bc_z, DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, DHMemory<float>& _bc_val_z, int3 _tile_dim, const cudaSurfaceObject_t& voxel_surface, const cudaSurfaceObject_t& velocity_surface, cudaStream_t _stream)
{
    uint8_t* is_bc_x    = _is_bc_x.dev_ptr_;
    uint8_t* is_bc_y    = _is_bc_y.dev_ptr_;
    uint8_t* is_bc_z    = _is_bc_z.dev_ptr_;
    float* bc_val_x  = _bc_val_x.dev_ptr_;
    float* bc_val_y  = _bc_val_y.dev_ptr_;
    float* bc_val_z  = _bc_val_z.dev_ptr_;
    SetBcBySurfaceKernel<<<Prod(_tile_dim), 128, 0, _stream>>>(is_bc_x, is_bc_y, is_bc_z, bc_val_x, bc_val_y, bc_val_z, _tile_dim, voxel_surface, velocity_surface);
}

__global__ void SetCoefByIsBcKernel(uint8_t* _is_dof, float* _a_diag, float* _a_x, float* _a_y, float* _a_z, int3 _tile_dim, const uint8_t* _is_bc_x, const uint8_t* _is_bc_y, const uint8_t* _is_bc_z)
{
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = t_id + i * 128;
        int idx        = tile_idx * 512 + voxel_idx;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int3 ijk       = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        int diag_cnt   = 6;
        // x-
        if (_is_bc_x[IjkToIdx(x_tile_dim, ijk)])
            diag_cnt--;
        // y-
        if (_is_bc_y[IjkToIdx(y_tile_dim, ijk)])
            diag_cnt--;
        // z-
        if (_is_bc_z[IjkToIdx(z_tile_dim, ijk)])
            diag_cnt--;
        // x+
        if (_is_bc_x[IjkToIdx(x_tile_dim, { ijk.x + 1, ijk.y, ijk.z })]) {
            diag_cnt--;
            _a_x[idx] = 0.0f;
        } else
            _a_x[idx] = -1.0f;
        // y+
        if (_is_bc_y[IjkToIdx(y_tile_dim, { ijk.x, ijk.y + 1, ijk.z })]) {
            diag_cnt--;
            _a_y[idx] = 0.0f;
        } else
            _a_y[idx] = -1.0f;
        // z+
        if (_is_bc_z[IjkToIdx(z_tile_dim, { ijk.x, ijk.y, ijk.z + 1 })]) {
            diag_cnt--;
            _a_z[idx] = 0.0f;
        } else
            _a_z[idx] = -1.0f;

        if (diag_cnt == 0)
            _is_dof[idx] = 0;
        else
            _is_dof[idx] = 1;
        _a_diag[idx] = (float)diag_cnt;
    }
}

void SetCoefByIsBcAsync(DHMemory<uint8_t>& _is_dof, DHMemory<float>& _a_diag, DHMemory<float>& _a_x, DHMemory<float>& _a_y, DHMemory<float>& _a_z, int3 _tile_dim, const DHMemory<uint8_t>& _is_bc_x, const DHMemory<uint8_t>& _is_bc_y, const DHMemory<uint8_t>& _is_bc_z, cudaStream_t _stream)
{
    uint8_t* is_dof        = _is_dof.dev_ptr_;
    float* a_diag       = _a_diag.dev_ptr_;
    float* a_x          = _a_x.dev_ptr_;
    float* a_y          = _a_y.dev_ptr_;
    float* a_z          = _a_z.dev_ptr_;
    const uint8_t* is_bc_x = _is_bc_x.dev_ptr_;
    const uint8_t* is_bc_y = _is_bc_y.dev_ptr_;
    const uint8_t* is_bc_z = _is_bc_z.dev_ptr_;
    int tile_num        = Prod(_tile_dim);
    SetCoefByIsBcKernel<<<tile_num, 128, 0, _stream>>>(is_dof, a_diag, a_x, a_y, a_z, _tile_dim, is_bc_x, is_bc_y, is_bc_z);
}

__device__ float InterpN2X(int3 _tile_dim, const float* _src, float3 _trans_pos, float _inv_dx)
{
    float eps               = 0.0001f;
    float3 ijk              = { _trans_pos.x * _inv_dx, _trans_pos.y * _inv_dx, _trans_pos.z * _inv_dx };
    float3 min_ijk          = { 0.5f + eps, 0.5f + eps, 0.5f + eps };
    float3 grid_dim         = { float(8 * _tile_dim.x), float(8 * _tile_dim.y), float(8 * _tile_dim.z) };
    float ret               = 0.0f;
    // x
    int3 axis_tile_dim      = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    float3 axis_ijk         = { ijk.x, ijk.y - 0.5f, ijk.z - 0.5f };
    float3 clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 0.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 1.5f - eps });
    int3 base_ijk           = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    float N2_x[3], N2_y[3], N2_z[3];
    int idx_x[3], idx_y[3], idx_z[3];

    int3 base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    int3 base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    int base_tile_idx  = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    int base_vol_idx   = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    int base_idx       = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _src[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    return ret;
}
__device__ float InterpN2Y(int3 _tile_dim, const float* _src, float3 _trans_pos, float _inv_dx)
{
    float eps       = 0.0001f;
    float3 ijk      = { _trans_pos.x * _inv_dx, _trans_pos.y * _inv_dx, _trans_pos.z * _inv_dx };
    float3 min_ijk  = { 0.5f + eps, 0.5f + eps, 0.5f + eps };
    float3 grid_dim = { float(8 * _tile_dim.x), float(8 * _tile_dim.y), float(8 * _tile_dim.z) };
    float ret       = 0.0f;

    int3 axis_tile_dim      = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    float3 axis_ijk         = { ijk.x - 0.5f, ijk.y, ijk.z - 0.5f };
    float3 clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 0.5f - eps, grid_dim.z - 1.5f - eps });
    int3 base_ijk           = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    float N2_x[3], N2_y[3], N2_z[3];
    int idx_x[3], idx_y[3], idx_z[3];

    int3 base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    int3 base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    int base_tile_idx  = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    int base_vol_idx   = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    int base_idx       = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _src[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    return ret;
}

__device__ float InterpN2Z(int3 _tile_dim, const float* _src, float3 _trans_pos, float _inv_dx)
{
    float eps       = 0.0001f;
    float3 ijk      = { _trans_pos.x * _inv_dx, _trans_pos.y * _inv_dx, _trans_pos.z * _inv_dx };
    float3 min_ijk  = { 0.5f + eps, 0.5f + eps, 0.5f + eps };
    float3 grid_dim = { float(8 * _tile_dim.x), float(8 * _tile_dim.y), float(8 * _tile_dim.z) };
    float ret       = 0.0f;

    int3 axis_tile_dim      = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    float3 axis_ijk         = { ijk.x - 0.5f, ijk.y - 0.5f, ijk.z };
    float3 clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 0.5f - eps });
    int3 base_ijk           = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    float N2_x[3], N2_y[3], N2_z[3];
    int idx_x[3], idx_y[3], idx_z[3];

    int3 base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    int3 base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    int base_tile_idx  = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    int base_vol_idx   = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    int base_idx       = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _src[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    return ret;
}

__device__ float InterpN2C(int3 _tile_dim, const float* _src, float3 _trans_pos, float _inv_dx)
{
    float eps       = 0.0001f;
    float3 ijk      = { _trans_pos.x * _inv_dx, _trans_pos.y * _inv_dx, _trans_pos.z * _inv_dx };
    float3 min_ijk  = { 0.5f + eps, 0.5f + eps, 0.5f + eps };
    float3 grid_dim = { float(8 * _tile_dim.x), float(8 * _tile_dim.y), float(8 * _tile_dim.z) };
    float ret       = 0.0f;

    int3 axis_tile_dim      = { _tile_dim.x, _tile_dim.y, _tile_dim.z };
    float3 axis_ijk         = { ijk.x - 0.5f, ijk.y - 0.5f, ijk.z - 0.5f };
    float3 clamped_axis_ijk = Clamp(axis_ijk, min_ijk, { grid_dim.x - 1.5f - eps, grid_dim.y - 1.5f - eps, grid_dim.z - 1.5f - eps });
    int3 base_ijk           = { int(clamped_axis_ijk.x - 0.5f), int(clamped_axis_ijk.y - 0.5f), int(clamped_axis_ijk.z - 0.5f) };
    float N2_x[3], N2_y[3], N2_z[3];
    int idx_x[3], idx_y[3], idx_z[3];

    int3 base_tile_ijk = { base_ijk.x >> 3, base_ijk.y >> 3, base_ijk.z >> 3 };
    int3 base_vol_ijk  = { base_ijk.x & 7, base_ijk.y & 7, base_ijk.z & 7 };
    int base_tile_idx  = base_tile_ijk.x * axis_tile_dim.y * axis_tile_dim.z + base_tile_ijk.y * axis_tile_dim.z + base_tile_ijk.z;
    int base_vol_idx   = (base_vol_ijk.x << 6) + (base_vol_ijk.y << 3) + base_vol_ijk.z;
    int base_idx       = (base_tile_idx << 9) + base_vol_idx;

    for (int i = 0; i < 3; i++) {
        float offset_x = clamped_axis_ijk.x - base_ijk.x - i;
        float offset_y = clamped_axis_ijk.y - base_ijk.y - i;
        float offset_z = clamped_axis_ijk.z - base_ijk.z - i;
        N2_x[i]        = N2(offset_x);
        N2_y[i]        = N2(offset_y);
        N2_z[i]        = N2(offset_z);
        idx_x[i]       = base_idx + (i << 6);
        idx_x[i] += ((base_vol_ijk.x + i) >> 3) ? (512 * axis_tile_dim.y * axis_tile_dim.z - 512) : 0;
        idx_y[i] = (i << 3);
        idx_y[i] += ((base_vol_ijk.y + i) >> 3) ? (512 * axis_tile_dim.z - 64) : 0;
        idx_z[i] = i;
        idx_z[i] += ((base_vol_ijk.z + i) >> 3) ? (512 - 8) : 0;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                int target_idx = idx_x[i] + idx_y[j] + idx_z[k];
                float val      = _src[target_idx];
                float N2_x_    = N2_x[i];
                float N2_y_    = N2_y[j];
                float N2_z_    = N2_z[k];
                ret += (val * N2_x_) * (N2_y_ * N2_z_);
            }
    return ret;
}

__global__ void RK2AdvectN2XKernel(float* _dst, int3 _tile_dim, const float* _src, const float* _u_x, const float* _u_y, const float* _u_z, float _dx, float _inv_dx, float _dt)
{
    int3 x_tile_dim = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    float half_dt   = 0.5f * _dt;
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(x_tile_dim, tile_idx);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx    = t_id + i * 128;
        int3 voxel_ijk   = VoxelIdxToIjk(voxel_idx);
        int3 ijk         = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float3 pos1      = { ijk.x * _dx, (ijk.y + 0.5f) * _dx, (ijk.z + 0.5f) * _dx };
        float3 u1        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos1, _inv_dx);
        float3 pos2      = { pos1.x - half_dt * u1.x, pos1.y - half_dt * u1.y, pos1.z - half_dt * u1.z };
        float3 u2        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos2, _inv_dx);
        float3 final_pos = { pos1.x - _dt * u2.x, pos1.y - _dt * u2.y, pos1.z - _dt * u2.z };
        int idx          = tile_idx * 512 + voxel_idx;
        _dst[idx]        = InterpN2X(_tile_dim, _src, final_pos, _inv_dx);
    }
}

__global__ void RK2AdvectN2YKernel(float* _dst, int3 _tile_dim, const float* _src, const float* _u_x, const float* _u_y, const float* _u_z, float _dx, float _inv_dx, float _dt)
{
    int3 y_tile_dim = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    float half_dt   = 0.5f * _dt;
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(y_tile_dim, tile_idx);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx    = t_id + i * 128;
        int3 voxel_ijk   = VoxelIdxToIjk(voxel_idx);
        int3 ijk         = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float3 pos1      = { (ijk.x + 0.5f) * _dx, ijk.y * _dx, (ijk.z + 0.5f) * _dx };
        float3 u1        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos1, _inv_dx);
        float3 pos2      = { pos1.x - half_dt * u1.x, pos1.y - half_dt * u1.y, pos1.z - half_dt * u1.z };
        float3 u2        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos2, _inv_dx);
        float3 final_pos = { pos1.x - _dt * u2.x, pos1.y - _dt * u2.y, pos1.z - _dt * u2.z };
        int idx          = tile_idx * 512 + voxel_idx;
        _dst[idx]        = InterpN2Y(_tile_dim, _src, final_pos, _inv_dx);
    }
}

__global__ void RK2AdvectN2ZKernel(float* _dst, int3 _tile_dim, const float* _src, const float* _u_x, const float* _u_y, const float* _u_z, float _dx, float _inv_dx, float _dt)
{
    int3 z_tile_dim = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    float half_dt   = 0.5f * _dt;
    int tile_idx    = blockIdx.x;
    int3 tile_ijk   = TileIdxToIjk(z_tile_dim, tile_idx);
    int t_id        = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx    = t_id + i * 128;
        int3 voxel_ijk   = VoxelIdxToIjk(voxel_idx);
        int3 ijk         = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float3 pos1      = { (ijk.x + 0.5f) * _dx, (ijk.y + 0.5f) * _dx, ijk.z * _dx };
        float3 u1        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos1, _inv_dx);
        float3 pos2      = { pos1.x - half_dt * u1.x, pos1.y - half_dt * u1.y, pos1.z - half_dt * u1.z };
        float3 u2        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos2, _inv_dx);
        float3 final_pos = { pos1.x - _dt * u2.x, pos1.y - _dt * u2.y, pos1.z - _dt * u2.z };
        int idx          = tile_idx * 512 + voxel_idx;
        _dst[idx]        = InterpN2Z(_tile_dim, _src, final_pos, _inv_dx);
    }
}

__global__ void RK2AdvectN2CKernel(float* _dst, int3 _tile_dim, const float* _src, const float* _u_x, const float* _u_y, const float* _u_z, float _dx, float _inv_dx, float _dt)
{
    float half_dt = 0.5f * _dt;
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx    = t_id + i * 128;
        int3 voxel_ijk   = VoxelIdxToIjk(voxel_idx);
        int3 ijk         = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
        float3 pos1      = { (ijk.x + 0.5f) * _dx, (ijk.y + 0.5f) * _dx, (ijk.z + 0.5f) * _dx };
        float3 u1        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos1, _inv_dx);
        float3 pos2      = { pos1.x - half_dt * u1.x, pos1.y - half_dt * u1.y, pos1.z - half_dt * u1.z };
        float3 u2        = InterpMacN2(_tile_dim, _u_x, _u_y, _u_z, pos2, _inv_dx);
        float3 final_pos = { pos1.x - _dt * u2.x, pos1.y - _dt * u2.y, pos1.z - _dt * u2.z };
        int idx          = tile_idx * 512 + voxel_idx;
        _dst[idx]        = InterpN2C(_tile_dim, _src, final_pos, _inv_dx);
    }
}

void AdvectN2XAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream)
{
    int3 x_tile_dim  = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    float* dst       = _dst.dev_ptr_;
    const float* src = _src.dev_ptr_;
    const float* u_x = _u_x.dev_ptr_;
    const float* u_y = _u_y.dev_ptr_;
    const float* u_z = _u_z.dev_ptr_;
    float inv_dx     = 1.0f / _dx;
    RK2AdvectN2XKernel<<<Prod(x_tile_dim), 128, 0, _stream>>>(dst, _tile_dim, src, u_x, u_y, u_z, _dx, inv_dx, _dt);
}

void AdvectN2YAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream)
{
    int3 y_tile_dim  = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    float* dst       = _dst.dev_ptr_;
    const float* src = _src.dev_ptr_;
    const float* u_x = _u_x.dev_ptr_;
    const float* u_y = _u_y.dev_ptr_;
    const float* u_z = _u_z.dev_ptr_;
    float inv_dx     = 1.0f / _dx;
    RK2AdvectN2YKernel<<<Prod(y_tile_dim), 128, 0, _stream>>>(dst, _tile_dim, src, u_x, u_y, u_z, _dx, inv_dx, _dt);
}

void AdvectN2ZAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream)
{
    int3 z_tile_dim  = { _tile_dim.x, _tile_dim.y, _tile_dim.z + 1 };
    float* dst       = _dst.dev_ptr_;
    const float* src = _src.dev_ptr_;
    const float* u_x = _u_x.dev_ptr_;
    const float* u_y = _u_y.dev_ptr_;
    const float* u_z = _u_z.dev_ptr_;
    float inv_dx     = 1.0f / _dx;
    RK2AdvectN2ZKernel<<<Prod(z_tile_dim), 128, 0, _stream>>>(dst, _tile_dim, src, u_x, u_y, u_z, _dx, inv_dx, _dt);
}
void AdvectN2CAsync(DHMemory<float>& _dst, int3 _tile_dim, const DHMemory<float>& _src, const DHMemory<float>& _u_x, const DHMemory<float>& _u_y, const DHMemory<float>& _u_z,
                    float _dx, float _dt, cudaStream_t _stream)
{
    float* dst       = _dst.dev_ptr_;
    const float* src = _src.dev_ptr_;
    const float* u_x = _u_x.dev_ptr_;
    const float* u_y = _u_y.dev_ptr_;
    const float* u_z = _u_z.dev_ptr_;
    float inv_dx     = 1.0f / _dx;
    RK2AdvectN2CKernel<<<Prod(_tile_dim), 128, 0, _stream>>>(dst, _tile_dim, src, u_x, u_y, u_z, _dx, inv_dx, _dt);
}

__global__ void SetInletXKernel(float* _bc_val_x, int3 _tile_dim, float _inlet_x)
{
    int tile_j           = blockIdx.x / _tile_dim.z;
    int tile_k           = blockIdx.x % _tile_dim.z;
    int voxel_j          = threadIdx.x / 8;
    int voxel_k          = threadIdx.x % 8;
    int3 left_ijk        = { 0, tile_j * 8 + voxel_j, tile_k * 8 + voxel_k };
    int3 right_ijk       = { 8 * _tile_dim.x, tile_j * 8 + voxel_j, tile_k * 8 + voxel_k };
    int3 x_tile_dim      = { _tile_dim.x + 1, _tile_dim.y, _tile_dim.z };
    int left_idx         = IjkToIdx(x_tile_dim, left_ijk);
    int right_idx        = IjkToIdx(x_tile_dim, right_ijk);
    _bc_val_x[left_idx]  = _inlet_x;
    _bc_val_x[right_idx] = _inlet_x;
}

__global__ void SetInletYKernel(float* _bc_val_y, int3 _tile_dim, float _inlet_y)
{
    int tile_i            = blockIdx.x / _tile_dim.z;
    int tile_j            = blockIdx.x % _tile_dim.z;
    int voxel_i           = threadIdx.x / 8;
    int voxel_k           = threadIdx.x % 8;
    int3 bottom_ijk       = { tile_i * 8 + voxel_i, 0, tile_j * 8 + voxel_k };
    int3 top_ijk          = { tile_i * 8 + voxel_i, 8 * _tile_dim.y, tile_j * 8 + voxel_k };
    int3 y_tile_dim       = { _tile_dim.x, _tile_dim.y + 1, _tile_dim.z };
    int bottom_idx        = IjkToIdx(y_tile_dim, bottom_ijk);
    int top_idx           = IjkToIdx(y_tile_dim, top_ijk);
    _bc_val_y[bottom_idx] = _inlet_y;
    _bc_val_y[top_idx]    = _inlet_y;
}
void SetInletAsync(DHMemory<float>& _bc_val_x, DHMemory<float>& _bc_val_y, int3 _tile_dim, float _inlet_norm, float _inlet_angle, cudaStream_t _stream)
{
    float* bc_val_x    = _bc_val_x.dev_ptr_;
    float* bc_val_y    = _bc_val_y.dev_ptr_;
    float radian_angle = _inlet_angle * 3.1415926f / 180.0f;
    float inlet_x      = _inlet_norm * cos(radian_angle);
    float inlet_y      = _inlet_norm * sin(radian_angle);
    SetInletXKernel<<<_tile_dim.y * _tile_dim.z, 64, 0, _stream>>>(bc_val_x, _tile_dim, inlet_x);
    SetInletYKernel<<<_tile_dim.x * _tile_dim.z, 64, 0, _stream>>>(bc_val_y, _tile_dim, inlet_y);
}
}