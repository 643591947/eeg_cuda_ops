import os
import tempfile
import torch
import numpy as np
import streamlit as st
import mne

from eeg_processor import (
    create_fir_weights,
    apply_fir_filter_cuda,
    apply_iir_filter_cuda,
    apply_fastica_cuda,
    get_data_segment,
    create_filtered_raw
)
from ui.sidebar import render_sidebar
from ui.plots import render_ica_topomaps

st.set_page_config(page_title="EEG CUDA Ops - Waveform Viewer", layout="wide")

st.title("🧠 EEG CUDA Ops Visualization Demo")
st.markdown("**Raw Waveform + FIR + IIR Filtering + CUDA FastICA**")

# 1. 渲染侧边栏，获取用户配置参数
cfg = render_sidebar()

# ==================== Main UI ====================
uploaded_file = st.file_uploader("Drag and drop your .bdf or .edf file here", type=["bdf", "edf"])

if uploaded_file is not None:
    original_ext = os.path.splitext(uploaded_file.name)[1].lower()
    st.session_state.original_extension = original_ext

    # 安全创建临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        with st.spinner("Loading EEG data..."):
            if original_ext == ".bdf":
                raw = mne.io.read_raw_bdf(tmp_path, preload=True, verbose=False)
            else:
                raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)

        total_duration = float(raw.times[-1]) if len(raw.times) > 0 else 0.0
        sfreq = raw.info['sfreq']

        st.success(f"✅ Load Successful! Format: {original_ext.upper()} | Sampling Rate: {sfreq:.1f} Hz | Total Duration: {total_duration:.2f} s")

        max_start = max(0.0, total_duration - 1.0)
        # 将 Start Time 滑块注入到侧边栏预留的占位符中
        start_time = cfg['start_time_placeholder'].slider("Start Time (s)", 0.0, float(max_start), value=0.0, step=0.5)

        picks = raw.ch_names[:cfg['n_channels']]
        plot_duration = min(cfg['duration'], total_duration - start_time)

        # ====================== Raw Waveform ======================
        st.subheader("📈 Raw EEG Waveform")
        with st.spinner("Generating raw waveform plot..."):
            fig_raw = raw.plot(picks=picks, duration=plot_duration, start=start_time,
                               n_channels=cfg['n_channels'], scalings={'eeg': cfg['amplitude_scale'] * 1e-6},
                               show=False, verbose=False, show_scrollbars=False)
            for ax in fig_raw.axes[:-1]: ax.grid(False)
            st.pyplot(fig_raw, use_container_width=True)

        # ====================== FIR Filtering ======================
        if cfg['apply_fir']:
            if cfg['fir_low'] >= cfg['fir_high']:
                st.error("❌ FIR Low cutoff must be lower than High cutoff!")
            else:
                with st.spinner(f"Applying FIR filter ({cfg['fir_low']}-{cfg['fir_high']} Hz)..."):
                    data, _ = get_data_segment(raw, picks, start_time, plot_duration)
                    weights, device = create_fir_weights(sfreq, cfg['fir_low'], cfg['fir_high'])
                    filtered_data = apply_fir_filter_cuda(data, weights, device)
                    raw_filtered = create_filtered_raw(raw, filtered_data, picks)

                    st.success(f"✅ FIR Filtering complete ({cfg['fir_low']}-{cfg['fir_high']} Hz)")

                    st.subheader(f"📈 FIR Filtered Waveform ({cfg['fir_low']}-{cfg['fir_high']} Hz)")
                    fig_filt = raw_filtered.plot(picks=picks, duration=plot_duration, start=0,
                                                 n_channels=cfg['n_channels'], scalings={'eeg': cfg['amplitude_scale'] * 1e-6},
                                                 show=False, verbose=False, show_scrollbars=False)
                    for ax in fig_filt.axes[:-1]: ax.grid(False)
                    st.pyplot(fig_filt, use_container_width=True)

        # ====================== IIR Filtering ======================
        if cfg['apply_iir']:
            with st.spinner(f"Applying IIR filter ({cfg['iir_type']})..."):
                data, _ = get_data_segment(raw, picks, start_time, plot_duration)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sfre_tensor = torch.tensor(sfreq, dtype=torch.float32).to(device)
                cfre_tensor = torch.tensor(cfg['iir_cfre'], dtype=torch.float32).to(device)

                bandstop = cfg['iir_bandstop'] if cfg['iir_type'] == "bandstop" else False

                filtered_data = apply_iir_filter_cuda(data, sfre_tensor, cfre_tensor, cfg['iir_type'], bandstop, version=cfg['iir_version'])
                raw_filtered = create_filtered_raw(raw, filtered_data, picks)

                cutoff_str = f"{cfg['iir_cfre'][0]:.1f}Hz" if len(cfg['iir_cfre']) == 1 else f"{cfg['iir_cfre'][0]:.1f}-{cfg['iir_cfre'][1]:.1f}Hz"
                st.success(f"✅ IIR ({cfg['iir_version']}) Filtering complete ({cfg['iir_type']} {cutoff_str})")

                st.subheader(f"📈 IIR Filtered Waveform ({cfg['iir_type']} {cutoff_str})")
                fig_iir = raw_filtered.plot(picks=picks, duration=plot_duration, start=0,
                                            n_channels=cfg['n_channels'], scalings={'eeg': cfg['amplitude_scale'] * 1e-6},
                                            show=False, verbose=False, show_scrollbars=False)
                for ax in fig_iir.axes[:-1]: ax.grid(False)
                st.pyplot(fig_iir, use_container_width=True)

        # ====================== FastICA Processing ======================
        if cfg['apply_ica']:
            with st.spinner("Executing CUDA Pipeline: Filtering -> Centering -> Whitening -> ICA..."):
                try:
                    # 1. 兼容性处理
                    if len(picks) > 0 and isinstance(picks[0], str):
                        selected_ch_names = picks
                        picks_idx = [raw.ch_names.index(name) for name in picks]
                    else:
                        selected_ch_names = [raw.ch_names[p] for p in picks]
                        picks_idx = picks

                    data, _ = get_data_segment(raw, picks_idx, start_time, plot_duration)

                    # 2. CUDA 滤波流水线逻辑 (保留原汁原味的提示信息)
                    if cfg['ica_pre_filter'] == "FIR":
                        st.info(f"🔄 Pipeline: Applying FIR Filter ({cfg['fir_low']}-{cfg['fir_high']}Hz) before ICA...")
                        weights, device = create_fir_weights(sfreq, cfg['fir_low'], cfg['fir_high'])
                        data = apply_fir_filter_cuda(data, weights, device)

                    elif cfg['ica_pre_filter'] in ["IIR (norm)", "IIR (matrix)"]:
                        iir_ver = "norm" if "norm" in cfg['ica_pre_filter'] else "matrix"
                        iir_cfre = cfg['iir_cfre']
                        iir_type = cfg['iir_type']
                        cutoff_str = f"{iir_cfre[0]:.1f}Hz" if len(iir_cfre) == 1 else f"{iir_cfre[0]:.1f}-{iir_cfre[1]:.1f}Hz"

                        st.info(f"🔄 Pipeline: Applying IIR Filter ({iir_type} {cutoff_str} | {iir_ver}) before ICA...")

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        sfre_tensor = torch.tensor(sfreq, dtype=torch.float32).to(device)
                        cfre_tensor = torch.tensor(iir_cfre, dtype=torch.float32).to(device)
                        bandstop = cfg['iir_bandstop'] if iir_type == "bandstop" else False

                        data = apply_iir_filter_cuda(data, sfre_tensor, cfre_tensor, iir_type, bandstop, version=iir_ver)

                    # 3. 执行 CUDA ICA 核心算法
                    real_tol = 10 ** (-cfg['ica_tol'])
                    ica_signals, w_whiten, w_ica = apply_fastica_cuda(data, max_iter=cfg['ica_max_iter'], tol=real_tol)

                    # 4. 计算全混合矩阵 A
                    W_full = np.dot(w_ica, w_whiten)
                    A = np.linalg.pinv(W_full)

                    # 缓存 session_state
                    st.session_state.ica_signals = ica_signals
                    st.session_state.A_matrix = A
                    st.session_state.ica_ch_names = selected_ch_names
                    st.session_state.ica_n_comps = ica_signals.shape[0]
                    st.session_state.ica_done = True

                    # ✅ 完美保留成功提示
                    st.success("✅ CUDA FastICA Pipeline Executed Successfully!")
                    st.subheader("🧩 CUDA ICA Spatial Components (Topomaps)")

                    info = raw.copy().pick_channels(selected_ch_names).info
                    montage = mne.channels.make_standard_montage('standard_1005')
                    info.set_montage(montage, match_case=False, on_missing='ignore')

                    # 调用解耦的绘图函数
                    render_ica_topomaps(A, info, ica_signals.shape[0])

                except Exception as e:
                    st.error(f"❌ ICA Pipeline Error: {str(e)}")
                    st.info("💡 建议：请确保选中的通道名称符合国际标准（如 Fp1, Fz）。")

        # ====================== ICA Rejection & Download ======================
        if st.session_state.get('ica_done', False):
            st.divider()
            st.subheader("🧹 Artifact Rejection")

            col1, col2 = st.columns([3, 1])
            with col1:
                exclude_idx_str = st.text_input("Enter component numbers to exclude (comma-separated, e.g. 1, 3)", value="")
            with col2:
                st.write("")
                st.write("")
                apply_reconstruction = st.button("🛠️ Reconstruct & Clean Data", type="primary", use_container_width=True)

            if apply_reconstruction:
                try:
                    if exclude_idx_str:
                        ica_signals_cached = st.session_state.ica_signals
                        A_cached = st.session_state.A_matrix
                        names_cached = st.session_state.ica_ch_names
                        n_comps_cached = st.session_state.ica_n_comps

                        exclude_indices = [int(i.strip()) - 1 for i in exclude_idx_str.split(',') if i.strip()]
                        valid_exclude = [i for i in exclude_indices if 0 <= i < n_comps_cached]

                        with st.spinner("Reconstructing clean data..."):
                            cleaned_ica_signals = ica_signals_cached.copy()
                            for idx in valid_exclude:
                                cleaned_ica_signals[idx, :] = 0

                            reconstructed_data_np = np.dot(A_cached, cleaned_ica_signals)
                            raw_cleaned = create_filtered_raw(raw, reconstructed_data_np, names_cached)

                            st.success(f"✅ Successfully excluded components: {[i + 1 for i in valid_exclude]}")

                            st.subheader("📈 Reconstructed (Cleaned) Waveform")
                            fig_clean = raw_cleaned.plot(duration=plot_duration, start=0,
                                                         n_channels=cfg['n_channels'],
                                                         scalings={'eeg': cfg['amplitude_scale'] * 1e-6},
                                                         show=False, verbose=False, show_scrollbars=False)
                            for ax in fig_clean.axes[:-1]: ax.grid(False)
                            st.pyplot(fig_clean, use_container_width=True)

                            # 安全生成文件用于下载
                            tmp_fd, safe_tmp_path = tempfile.mkstemp(suffix=".fif")
                            os.close(tmp_fd)
                            try:
                                raw_cleaned.save(safe_tmp_path, overwrite=True, verbose=False)
                                with open(safe_tmp_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Download Cleaned EEG (.fif)",
                                        data=file.read(),
                                        file_name="cleaned_eeg_data.fif",
                                        mime="application/octet-stream",
                                        use_container_width=True
                                    )
                            finally:
                                if os.path.exists(safe_tmp_path):
                                    os.remove(safe_tmp_path)
                    else:
                        st.warning("⚠️ Please enter at least one component number to exclude.")

                except Exception as e:
                    st.error(f"❌ Processing Error: {str(e)}")

    except Exception as e:
        st.error(f"❌ File Processing Error: {str(e)}")
    finally:
        # 清理用户最初上传生成的临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Please upload a .bdf or .edf file to view waveforms")

st.caption("Powered by MNE-Python + Streamlit + eeg_cuda")