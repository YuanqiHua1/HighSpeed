import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, Edge
from scipy.io import loadmat

# ==============================
# Parameters
# ==============================
DEVICE = "Dev1"
Rate = 50000

mat = loadmat('singlehigh.mat')
stim = np.squeeze(mat['singlehigh']['totalwave'])  # 取出刺激波形
shock_fraction = 1.0  # 这里改成你想要的倍率，比如0.5降低一半幅度
stim = stim * shock_fraction
print(f'Max stimulus amplitude after scaling: {np.max(np.abs(stim))} V')
num_samples = len(stim)
stim_duration = num_samples / Rate  # 刺激时长秒

# 构造数字输出数据 (2线：Aim In 和 Trigger)
digital_data = np.zeros((num_samples, 2), dtype=bool)

# Aim In 信号全程高电平（预备状态）
digital_data[:, 0] = True

# Trigger脉冲在1秒处开始，宽度100ms (根据需要改)
trigger_start = int(Rate * 1.0)
trigger_width = int(Rate * 0.1)
digital_data[trigger_start:trigger_start+trigger_width, 1] = True

with nidaqmx.Task() as ao_task, nidaqmx.Task() as do_task:

    # 模拟输出，电刺激波形
    ao_task.ao_channels.add_ao_voltage_chan(f"{DEVICE}/ao0", min_val=-5, max_val=5)
    ao_task.timing.cfg_samp_clk_timing(Rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=num_samples)

    # 数字输出，Aim In 和 Trigger
    do_task.do_channels.add_do_chan(f"{DEVICE}/port0/line0:1")
    do_task.timing.cfg_samp_clk_timing(Rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=num_samples)

    # 设置外部触发，等待PFI0上升沿启动
    ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(f'/{DEVICE}/PFI0', trigger_edge=Edge.RISING)
    do_task.triggers.start_trigger.cfg_dig_edge_start_trig(f'/{DEVICE}/PFI0', trigger_edge=Edge.RISING)

    # 写入数据，但不自动开始 （注意：auto_start=False，等待同时启动）
    ao_task.write(stim, auto_start=False)
    do_task.write(digital_data, auto_start=False)

    # 启动任务
    do_task.start() # 先启动数字输出，保持Aim In状态，等待触发
    ao_task.start() # 启动模拟输出（电刺激波形），等待触发
    print("Start! Waiting FasMotion Arm In ")

    # 等待模拟输出完成
    ao_task.wait_until_done(timeout=stim_duration + 5)

print("Done!")