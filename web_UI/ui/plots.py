import streamlit as st
from matplotlib import pyplot as plt
import mne
import numpy as np
import io
from PIL import Image

def render_ica_topomaps(A_matrix, info, n_comps):
    """绘制 ICA 独立成分的头皮地形图"""
    cols = 5
    rows = int(np.ceil(n_comps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    if rows == 1:
        axes = np.expand_dims(axes, 0)
    axes = axes.flatten()

    for i in range(n_comps):
        mne.viz.plot_topomap(
            A_matrix[:, i],
            info,
            axes=axes[i],
            show=False,
            extrapolate='head',
            outlines='head',
        )
        axes[i].set_title(f'ICA {i + 1}', fontsize=10)

    # 隐藏多余的空白子图
    for i in range(n_comps, len(axes)):
        axes[i].axis('off')

    fig.tight_layout()

    # 将结果转为图片渲染到 Streamlit
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.image(Image.open(buf), caption="Aligned CUDA ICA Components", use_container_width=True)

    # 清理内存
    plt.close(fig)
    buf.close()