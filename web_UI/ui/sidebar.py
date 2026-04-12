import streamlit as st


def render_sidebar():
    """渲染侧边栏并返回所有配置参数"""
    params = {}

    with st.sidebar:
        st.header("Processing Settings")

        # --- FIR ---
        st.subheader("🔧 FIR Filter")
        params['fir_low'] = st.number_input("FIR Low Cutoff (Hz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        params['fir_high'] = st.number_input("FIR High Cutoff (Hz)", min_value=0.1, max_value=100.0, value=40.0,
                                             step=0.1)
        params['apply_fir'] = st.button("🚀 Apply FIR Filter", type="primary", use_container_width=True)
        st.divider()

        # --- IIR ---
        st.subheader("📊 IIR Filter")
        params['iir_version'] = st.selectbox("IIR Implementation", ["norm", "matrix"], index=0)
        params['iir_type'] = st.selectbox("Filter Type", ["low", "high", "bandpass", "bandstop"], index=0)

        if params['iir_type'] in ["low", "high"]:
            iir_cutoff = st.number_input("Cutoff Frequency (Hz)", min_value=0.1, max_value=100.0, value=40.0, step=0.1)
            params['iir_cfre'] = [iir_cutoff]
        else:
            iir_low = st.number_input("Low Cutoff (Hz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            iir_high = st.number_input("High Cutoff (Hz)", min_value=0.1, max_value=100.0, value=40.0, step=0.1)
            params['iir_cfre'] = [iir_low, iir_high]

        params['iir_bandstop'] = st.checkbox("Enable Bandstop", value=False)
        params['apply_iir'] = st.button("📊 Apply IIR Filter", type="primary", use_container_width=True)
        st.divider()

        # --- ICA ---
        st.subheader("🧩 FastICA Analysis")
        params['ica_pre_filter'] = st.selectbox(
            "Pre-ICA Filter", ["None", "FIR", "IIR (norm)", "IIR (matrix)"], index=1,
            help="建议在 ICA 前进行 1Hz 以上高通或带通滤波以消除基线漂移"
        )
        params['ica_max_iter'] = st.number_input("Max Iterations", min_value=50, max_value=1000, value=200, step=50)
        params['ica_tol'] = st.number_input("Tolerance (1e-X)", min_value=2, max_value=6, value=4, step=1)
        params['apply_ica'] = st.button("🧩 Apply FastICA Pipeline", type="primary", use_container_width=True)
        st.divider()

        # --- Display ---
        params['duration'] = st.slider("Display Duration (s)", min_value=1.0, max_value=120.0, value=20.0, step=1.0)
        params['n_channels'] = st.slider("Number of Channels", 4, 64, 32, step=2)
        params['amplitude_scale'] = st.slider("Amplitude Scale (μV/div)", 20, 200, 40, step=5)

        # 将 start_time 占位，稍后在主函数中根据 total_duration 更新
        params['start_time_placeholder'] = st.empty()

    return params