import nidaqmx
import numpy as np
import time
from scipy.io import loadmat

# 载入.mat文件中的刺激波形
mat = loadmat('singlehigh.mat')
totalwave = mat['singlehigh']['totalwave'][0,0].flatten()  # 根据.mat结构调整

# 参数
rate = 50000  # 采样率
num_samples = len(totalwave)

# 生成数字信号数组
digital_data = np.zeros((num_samples, 2), dtype=bool)
digital_data[:, 0] = True  # Arm In 全程高电平

trigger_start = int(rate * 1.0)     # 触发点：1秒
trigger_duration = int(rate * 0.1)  # 触发持续100ms
digital_data[trigger_start:trigger_start+trigger_duration, 1] = True  # Trigger脉冲

# 创建NI-DAQ任务
with nidaqmx.Task() as task:
    # 模拟输出通道（电刺激信号）
    task.ao_channels.add_ao_voltage_chan("Dev1/ao0")

    # 数字输出通道（Arm In和Trigger）
    task.do_channels.add_do_chan("Dev1/port0/line0:1")

    # 设置采样时钟
    task.timing.cfg_samp_clk_timing(rate,
                                   sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                   samps_per_chan=num_samples)

    # 组合模拟和数字信号写入（先分开写，后面有高级方式同步写）
    # 这里示范同步写两个信号，需要用write()的字典格式
    task.write({
        'Dev1/ao0': totalwave.tolist(),
        'Dev1/port0/line0:1': digital_data.tolist()
    }, auto_start=False)

    # 启动任务
    task.start()

    # 等待任务完成
    task.wait_until_done(timeout=10)

    # 任务停止自动执行
    task.stop()

print("Done")


"""
运行前注意：

替换 "Dev1/ao0" 和 "Dev1/port0/line0:1" 为你设备的实际端口名

确认NI-DAQmx驱动已安装且nidaqmx Python库可用（pip install nidaqmx）

你的singlehigh.mat文件结构要和上面loadmat读取代码匹配，或者调整flatten()部分

触发时间和宽度可以根据需要调整trigger_start和trigger_duration变量
"""