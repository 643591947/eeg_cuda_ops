import streamlit as st
import mne
import tempfile
import os
import torch

from eeg_processor import (
    create_fir_weights,
    apply_fir_filter_cuda,
    apply_iir_filter_cuda,
    get_data_segment,
    create_filtered_raw
)

# streamlit run main.py

st.set_page_config(page_title="EEG CUDA Ops - Waveform Viewer", layout="wide")

st.title("🧠 EEG CUDA Ops Visualization Demo")
st.markdown("**Raw Waveform + FIR + IIR Filtering**")

# ==================== Sidebar ====================
with st.sidebar:
    st.header("Filter Settings")

    st.subheader("🔧 FIR Filter")
    fir_low = st.number_input("FIR Low Cutoff (Hz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    fir_high = st.number_input("FIR High Cutoff (Hz)", min_value=0.1, max_value=100.0, value=40.0, step=0.1)
    apply_fir = st.button("🚀 Apply FIR Filter", type="primary", use_container_width=True)

    st.divider()

    st.subheader("📊 IIR Filter")
    iir_version = st.selectbox("IIR Implementation", ["norm", "matrix"], index=0)
    iir_type = st.selectbox("Filter Type", ["low", "high", "bandpass", "bandstop"], index=0)

    if iir_type in ["low", "high"]:
        iir_cutoff = st.number_input("Cutoff Frequency (Hz)", min_value=0.1, max_value=100.0, value=40.0, step=0.1)
        iir_cfre = [iir_cutoff]
    else:
        iir_low = st.number_input("Low Cutoff (Hz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
        iir_high = st.number_input("High Cutoff (Hz)", min_value=0.1, max_value=100.0, value=40.0, step=0.1)
        iir_cfre = [iir_low, iir_high]

    iir_bandstop = st.checkbox("Enable Bandstop", value=False)
    apply_iir = st.button("📊 Apply IIR Filter", type="primary", use_container_width=True)

    st.divider()

    duration = st.slider("Display Duration (s)", min_value=1.0, max_value=120.0, value=20.0, step=1.0)
    n_channels = st.slider("Number of Channels", 4, 64, 32, step=2)
    amplitude_scale = st.slider("Amplitude Scale (μV/div)", 20, 200, 40, step=5)

# ==================== Main UI ====================
uploaded_file = st.file_uploader("Drag and drop your .bdf or .edf file here", type=["bdf", "edf"])

if uploaded_file is not None:
    original_ext = os.path.splitext(uploaded_file.name)[1].lower()
    st.session_state.original_extension = original_ext

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

        st.success(
            f"✅ Load Successful! Format: {original_ext.upper()} | Sampling Rate: {sfreq:.1f} Hz | Total Duration: {total_duration:.2f} s")

        max_start = max(0.0, total_duration - 1.0)
        start_time = st.sidebar.slider("Start Time (s)", 0.0, float(max_start), value=0.0, step=0.5)

        picks = raw.ch_names[:n_channels]
        plot_duration = min(duration, total_duration - start_time)

        # ====================== Raw Waveform ======================
        st.subheader("📈 Raw EEG Waveform")
        with st.spinner("Generating raw waveform plot..."):
            fig_raw = raw.plot(picks=picks, duration=plot_duration, start=start_time,
                               n_channels=n_channels, scalings={'eeg': amplitude_scale * 1e-6},
                               show=False, verbose=False, show_scrollbars=False, block=False)
            for ax in fig_raw.axes[:-1]: ax.grid(False)
            st.pyplot(fig_raw, use_container_width=True)

        # ====================== FIR Filtering ======================
        if apply_fir:
            if fir_low >= fir_high:
                st.error("❌ FIR Low cutoff must be lower than High cutoff!")
            else:
                with st.spinner(f"Applying FIR filter ({fir_low}-{fir_high} Hz)..."):
                    data, _ = get_data_segment(raw, picks, start_time, plot_duration)
                    weights, device = create_fir_weights(sfreq, fir_low, fir_high)
                    filtered_data = apply_fir_filter_cuda(data, weights, device)
                    raw_filtered = create_filtered_raw(raw, filtered_data, picks)

                    st.success(f"✅ FIR Filtering complete ({fir_low}-{fir_high} Hz)")

                    st.subheader(f"📈 FIR Filtered Waveform ({fir_low}-{fir_high} Hz)")
                    fig_filt = raw_filtered.plot(picks=picks, duration=plot_duration, start=0,
                                                 n_channels=n_channels, scalings={'eeg': amplitude_scale * 1e-6},
                                                 show=False, verbose=False, show_scrollbars=False, block=False)
                    for ax in fig_filt.axes[:-1]: ax.grid(False)
                    st.pyplot(fig_filt, use_container_width=True)

        # ====================== IIR Filtering ======================
        if apply_iir:
            with st.spinner(f"Applying IIR filter ({iir_type})..."):
                data, _ = get_data_segment(raw, picks, start_time, plot_duration)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sfre_tensor = torch.tensor(sfreq, dtype=torch.float32).to(device)
                cfre_tensor = torch.tensor(iir_cfre, dtype=torch.float32).to(device)

                pass_type = iir_type
                bandstop = iir_bandstop if iir_type == "bandstop" else False

                filtered_data = apply_iir_filter_cuda(data, sfre_tensor, cfre_tensor, pass_type, bandstop,
                                                      version=iir_version)
                raw_filtered = create_filtered_raw(raw, filtered_data, picks)

                cutoff_str = f"{iir_cfre[0]:.1f}Hz" if len(iir_cfre) == 1 else f"{iir_cfre[0]:.1f}-{iir_cfre[1]:.1f}Hz"
                st.success(f"✅ IIR ({iir_version}) Filtering complete ({iir_type} {cutoff_str})")

                st.subheader(f"📈 IIR Filtered Waveform ({iir_type} {cutoff_str})")
                fig_iir = raw_filtered.plot(picks=picks, duration=plot_duration, start=0,
                                            n_channels=n_channels, scalings={'eeg': amplitude_scale * 1e-6},
                                            show=False, verbose=False, show_scrollbars=False, block=False)
                for ax in fig_iir.axes[:-1]: ax.grid(False)
                st.pyplot(fig_iir, use_container_width=True)

        with st.expander("📋 Full Channel List"):
            st.write(raw.ch_names)

    except Exception as e:
        st.error(f"❌ Processing Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Please upload a .bdf or .edf file to view waveforms")

st.caption("Powered by MNE-Python + Streamlit + eeg_cuda")