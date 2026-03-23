import cv2
import numpy as np
import matplotlib.pyplot as plt

from videowrapper import VideoWrapper


clicked_point = None


def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked point: {clicked_point}")


def visualize_arc_search(
    avi_path,
    frame_id=50,
    arcradius=30,
    n_samples=100,
    guess_vector=(0, 1)
):
    global clicked_point

    vid = VideoWrapper(avi_path)

    # 取指定帧
    vid.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = vid.cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_id}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- 让用户点 current ----------
    cv2.namedWindow("Click a tail point (press ESC when done)")
    cv2.setMouseCallback("Click a tail point (press ESC when done)", on_mouse)

    temp = frame.copy()
    print("Click ONE point on the tail, then press ESC")

    while True:
        display = temp.copy()
        if clicked_point is not None:
            cv2.circle(display, clicked_point, 4, (0, 0, 255), -1)
        cv2.imshow("Click a tail point (press ESC when done)", display)
        key = cv2.waitKey(20)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

    if clicked_point is None:
        raise RuntimeError("No point clicked")

    current = np.array(clicked_point, dtype=float)

    # ---------- 画 arcradius 搜索弧 ----------
    guess_vector = np.array(guess_vector, dtype=float)
    guess_vector = guess_vector / np.linalg.norm(guess_vector)

    angles = np.linspace(-np.pi / 4, np.pi / 4, n_samples)
    arc_points = []
    intensities = []

    for a in angles:
        rot = np.array([
            [np.cos(a), -np.sin(a)],
            [np.sin(a),  np.cos(a)]
        ])
        direction = rot @ guess_vector
        p = current + arcradius * direction
        x, y = int(p[0]), int(p[1])

        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            arc_points.append((x, y))
            intensities.append(gray[y, x])
        else:
            arc_points.append((x, y))
            intensities.append(255)

    intensities = np.array(intensities)
    best_idx = np.argmin(intensities)
    best_point = arc_points[best_idx]

    # ---------- 图 1：画在图像上 ----------
    vis = frame.copy()

    cv2.circle(vis, tuple(current.astype(int)), 5, (0, 0, 255), -1)   # current
    for p in arc_points:
        cv2.circle(vis, p, 1, (255, 0, 0), -1)                        # arc
    cv2.circle(vis, best_point, 6, (0, 255, 0), -1)                  # selected

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Frame {frame_id} | arcradius={arcradius}")
    plt.axis("off")

    # ---------- 图 2：强度曲线 ----------
    plt.figure(figsize=(6, 3))
    plt.plot(intensities, label="Intensity (lower = darker)")
    plt.scatter(best_idx, intensities[best_idx], color="red", label="Selected")
    plt.xlabel("Arc sample index")
    plt.ylabel("Gray value")
    plt.legend()
    plt.title("Intensity along arc")

    plt.show()


if __name__ == "__main__":
    visualize_arc_search(
        avi_path=r"U:\YuanqiHua\High speed\260127 dmrt3 MTZ\AVI_out\d3-MTZ-1_1_0.avi",
        frame_id=150,        # ← 你可以改
        arcradius=60,       # ← 30 / 60 对比
        guess_vector=(-1, 0) # ← 大概沿尾巴方向 （-1，0）右，（1，0）左，（0，1）下，（0，-1）上，
    )
