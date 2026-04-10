![EEG CUDA 运算](images/the_banner.png)
# EEG_CUDA：高性能脑电信号处理加速器

[ English | [中文](./README_zh.md) ]

EEG_CUDA 是一个专为脑电图（EEG）信号处理设计的高性能库。通过利用 NVIDIA CUDA 技术，本项目在保持高数值稳定性的同时，加速了长序列、多通道数据的关键算法。

数学原理详见 `docs/math.md`。

---
## 📊 性能基准测试

我们使用以下配置在典型 EEG 数据规模上进行了全面的基准测试：
