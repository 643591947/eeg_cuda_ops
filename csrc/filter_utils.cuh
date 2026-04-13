// filter_utils.cuh
#pragma once

#include <torch/extension.h>
#include <vector>
#include <utility>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * 计算数字滤波器中的归一化角频率 Omega（预扭曲频率）
 *
 * @param sfre  采样频率 (Sampling Frequency)
 * @param cfre  截止频率 (Cutoff Frequency)
 * @return      预扭曲后的角频率 ω = tan(π * cfre / sfre)
 */
static inline float Omega(float sfre, float cfre) {
    return tanf(M_PI * cfre / sfre);
}

/**
 * 计算二阶 IIR 滤波器系数 a、b（Butterworth 风格）
 *
 * 支持两种模式：
 *   1. 单截止频率：低通 (low) 或 高通 (high)
 *   2. 双截止频率：带通 或 带阻 (bandstop)
 *
 * @param sfre     采样频率张量（标量）
 * @param cfre     截止频率张量（标量或含2个元素的1D张量）
 * @param pass     当 cfre 为单值时使用："low" 或 "high"
 * @param bandstop 当 cfre 为两个值时，true 表示带阻滤波器
 * @return         std::pair<b, a>，每个都是长度为3的vector<float>
 */
static inline std::pair<std::vector<float>, std::vector<float>> a_b(
    torch::Tensor sfre,
    torch::Tensor cfre,
    const std::string& pass = "low",   // 'low' or 'high'
    bool bandstop = false
) {
    TORCH_CHECK(sfre.numel() == 1, "Sampling frequency (sfre) must be a scalar tensor");
    TORCH_CHECK(cfre.is_cpu() || cfre.is_cuda(), "cfre must be a tensor");

    std::vector<float> b(3, 0.0f);
    std::vector<float> a(3, 0.0f);

    float sfre_val = sfre.item<float>();

    if (cfre.numel() == 2) {
        // ====================== 带通 / 带阻滤波器 ======================
        TORCH_CHECK(cfre.dim() == 1, "Cutoff frequency array must be 1-dimensional");

        float f1 = cfre[0].item<float>();
        float f2 = cfre[1].item<float>();
        if (f1 > f2) std::swap(f1, f2);

        float low_omega  = Omega(sfre_val, f1);
        float high_omega = Omega(sfre_val, f2);
        float BW         = high_omega - low_omega;
        float omega_2    = low_omega * high_omega;

        float a0_inv = 1.0f / (1.0f + BW + omega_2);

        a[0] = 1.0f;
        a[1] = 2.0f * (omega_2 - 1.0f) * a0_inv;
        a[2] = (1.0f - BW + omega_2) * a0_inv;

        if (bandstop) {
            // Band-stop filter
            b[0] =  BW * a0_inv;
            b[1] =  0.0f;
            b[2] = -BW * a0_inv;
        } else {
            // Band-pass filter
            b[0] = (1.0f + omega_2) * a0_inv;
            b[1] = 2.0f * (omega_2 - 1.0f) * a0_inv;
            b[2] = (1.0f + omega_2) * a0_inv;
        }

    } else if (cfre.numel() == 1) {
        // ====================== 低通 / 高通滤波器 ======================
        float omega    = Omega(sfre_val, cfre[0].item<float>());
        float sqrt2    = std::sqrt(2.0f);
        float omega_sq = omega * omega;

        float a0_inv = 1.0f / (1.0f + sqrt2 * omega + omega_sq);

        a[0] = 1.0f;
        a[1] = 2.0f * (omega_sq - 1.0f) * a0_inv;
        a[2] = (1.0f - sqrt2 * omega + omega_sq) * a0_inv;

        if (pass == "low") {
            // Low-pass filter
            b[0] = omega_sq * a0_inv;
            b[1] = 2.0f * omega_sq * a0_inv;
            b[2] = omega_sq * a0_inv;
        } else if (pass == "high") {
            // High-pass filter
            b[0] =  1.0f * a0_inv;
            b[1] = -2.0f * a0_inv;
            b[2] =  1.0f * a0_inv;
        } else {
            TORCH_CHECK(false, "pass must be 'low' or 'high' when using single cutoff frequency");
        }

    } else {
        TORCH_CHECK(false,
            "cfre must have either 1 element (low/highpass) or 2 elements (bandpass/bandstop), got ",
            cfre.numel(), " elements");
    }

    return {b, a};
}