import time

from dataCleaner import *

input_dir = r"E:\bizq\脑电分析\EEG测试数据\Copy_20260130175721005\赵婷\赵婷_20260130_173432\1\1\data.bdf"
raw_data = mne.io.read_raw_bdf(input_dir, preload=True)


# 复制 50 份并拼接
raw_list = [raw_data.copy() for _ in range(100)]
massive_raw = mne.concatenate_raws(raw_list)

print(f"当前测试数据总时长: {massive_raw.times[-1] / 60:.2f} 分钟")
print(f"数据矩阵形状: {massive_raw.get_data().shape}")

cleaner = DataCleaner()

# 开始计时
start_time = time.time()
clean_data = cleaner.clean_data(raw_data)

print(f"GPU 清洗 {massive_raw.times[-1] / 3600:.2f} 小时脑电数据总耗时: {time.time() - start_time:.3f} 秒")
