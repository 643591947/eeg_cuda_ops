import scipy
import torch
import numpy as np
import mne
import eeg_cuda


class DataCleaner:
    def __init__(self):
        pass

    def clean_data(self, data_raw, weights=None):
        data_raw = data_raw
        weights_tensor = weights

        # ==========================================
        # 0. 数据提取与预处理
        # ==========================================
        # 提取 numpy 数据并保证是 float32
        np_data = data_raw.get_data().astype(np.float32)
        fs = data_raw.info['sfreq']

        # 处理权重逻辑
        if weights_tensor is None:
            weights_np = scipy.signal.firwin(255, [1, 40], pass_zero=False, fs=fs).astype(np.float32)
            weights_tensor = torch.from_numpy(weights_np).cuda()
        else:
            # 确保传入的 weights 在 GPU 上且为 float32
            if not isinstance(weights_tensor, torch.Tensor):
                weights_tensor = torch.tensor(weights_tensor)
            weights_tensor = weights_tensor.float().cuda()

        # 开启无梯度模式
        with torch.no_grad():
            eeg_tensor = torch.from_numpy(np_data).unsqueeze(0).cuda()

            # ==========================================
            # 1. 滤波 + 中心化
            # ==========================================
            print("正在执行 CUDA 滤波与中心化......")
            filtered = eeg_cuda.fir_filter(eeg_tensor, weights_tensor)
            mean_val = filtered.mean(dim=-1, keepdim=True)
            centered_eeg = eeg_cuda.centering(filtered)

            # ==========================================
            # 2. 白化
            # ==========================================
            print("正在执行 CUDA 白化......")
            whitened_eeg, p_cuda_tensor = eeg_cuda.whitening(centered_eeg)

            # ==========================================
            # 3. CUDA FastICA
            # ==========================================
            print("正在执行 CUDA FastICA 迭代......")
            S_cuda_tensor, W_cuda_tensor = eeg_cuda.fastica_iter(whitened_eeg, max_iter=200, tol=1e-4)

            # ==========================================
            # 4. 伪影剔除 (升级为峰度 Kurtosis 识别)
            # ==========================================
            print("正在执行 基于峰度的伪影剔除......")
            S_clean = S_cuda_tensor.clone()

            # 计算均值和方差
            mean_S = S_clean.mean(dim=-1, keepdim=True)
            var_S = S_clean.var(dim=-1, keepdim=True)

            # 计算峰度: E[(x-\mu)^4] / \sigma^2
            # 峰度对突发性的巨大脉冲（眨眼）极其敏感
            kurtosis = ((S_clean - mean_S) ** 4).mean(dim=-1, keepdim=True) / (var_S ** 2)
            kurtosis = kurtosis.squeeze()  # 形状变为 [Channels]

            # 找出峰度最大的成分干掉
            blink_idx = torch.argmax(kurtosis, dim=0)
            print(f" -> 剔除疑似眨眼成分: Index {blink_idx.item()}")
            S_clean[0, blink_idx, :] = 0.0

            # ==========================================
            # 5. 逆投影重建回物理通道
            # ==========================================
            print("正在执行 逆投影重建......")
            inverse_matrix = torch.linalg.inv(torch.bmm(W_cuda_tensor, p_cuda_tensor))
            clean_data_gpu = torch.bmm(inverse_matrix, S_clean) + mean_val

            # ==========================================
            # 6. 导出
            # ==========================================
            clean_data_np = clean_data_gpu.squeeze(0).cpu().numpy()

        # 7. 创建一个新的 RawArray 并保留原有 info 信息
        info = data_raw.info
        clean_raw = mne.io.RawArray(clean_data_np, info)

        return clean_raw