import tailclick
from videowrapper import VideoWrapper
import jtailfit2 #as jtailfit2  # ✅ 用你已改好的 jtailfit2.py
import pickle
import os
import os.path
import numpy as np

START_INDEX = 96  # 从第几个开始（1-based），想从第12个就写 12


if __name__ == "__main__":
    input_folder = r'\\Hive3014\znn\YuanqiHua\High speed\260212 MTZ control\AVI_out_binary'
    output_folder = r'\\Hive3014\znn\YuanqiHua\High speed\260212 MTZ control\pkl_output'
    os.makedirs(output_folder, exist_ok=True)

    # ✅ 排序 + 打印序号（很少改动）
    avi_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('.avi')])
    for i, name in enumerate(avi_files, start=1):
        print(f"[{i:03d}] {name}")

    for i, avi_name in enumerate(avi_files[START_INDEX-1:], start=START_INDEX):
        filepath = os.path.join(input_folder, avi_name)
        print(f'\nProcessing [{i:03d}/{len(avi_files):03d}] {avi_name}')

        vid = VideoWrapper(filepath)
        firstframe = vid.firstframe

        startpoint, endpoint = tailclick.picktwopoints(firstframe)

        """
        # 旧代码
        tf, stop_after_video = jtailfit2.tailfit_simple(
            vid, startpoint, endpoint,
            display=True,
            # arcradius=30,
            variabledelay=False,  # ✅ 想更快更稳定建议 False
            fixed_delay_ms=1,     # ✅ 越小越快
            speed=5,# ✅ 倍速：你可调 3~10
            point_color=(0,255,0), point_radius=4,
            halfwidth = 18,
            min_width = 5
        )
        """
        tf, stop_after_video = jtailfit2.tailfit_skeleton(
            vid,
            startpoint,
            endpoint,
            num_points=20,
            display=True,
            speed=5,
            roi_thickness=100, # tail size
            roi_dilate=8,
            min_mask_area=500,
        )

        # Save output PKL
        pkl_name = os.path.splitext(avi_name)[0] + '.pkl'
        pkl_path = os.path.join(output_folder, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(tf, f)

        # ✅ 如果你按了 x：处理完当前视频后停止整个批处理
        if stop_after_video:
            print("Stop requested (pressed x). Stopping batch.")
            break

    print('All videos processed.')


