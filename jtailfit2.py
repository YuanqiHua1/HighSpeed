import os
def roi_enhance_tail(frame_gray, start, end):
    """
    方法一：尾巴区域加深，背景减弱
    frame_gray: 单通道灰度图 uint8
    start, end: 尾根和尾尖坐标（np.array）
    """
    import numpy as np
    import cv2


    g = frame_gray.astype(np.float32)

    # 1) 背景整体变亮（趋近白）
    bg = g * 0.7 + 80   # 可调：0.6~0.9, 40~120

    # 2) 尾巴整体加深
    tail = g * 0.45    # 可调：0.3~0.6

    # 3) 画“尾巴长条 ROI”
    mask = np.zeros_like(g, dtype=np.uint8)
    p1 = tuple(start.astype(int))
    p2 = tuple(end.astype(int))

    thickness = 50     # ⭐ 很关键：尾巴区域宽度（30~60 常用）
    cv2.line(mask, p1, p2, 255, thickness)

    # 4) 合成：ROI 内用 tail，外面用 bg
    out = bg.copy()
    out[mask > 0] = tail[mask > 0]

    return np.clip(out, 0, 255).astype(np.uint8)


def getbackgroundsign(frame, start_point):
    """
    用“尾根小块 vs 尾根周围大块背景”的均值对比，决定尾巴相对背景是更暗还是更亮。
    返回 +1 或 -1，用于把尾巴变成“更容易被选到”的方向。
    """
    import numpy as np
    x, y = int(start_point[0]), int(start_point[1])

    r_tail = 5   # 尾根小块
    r_bg   = 30  # 周围背景大块

    ys1 = slice(max(0, y - r_tail), min(frame.shape[0], y + r_tail))
    xs1 = slice(max(0, x - r_tail), min(frame.shape[1], x + r_tail))
    tail_local = frame[ys1, xs1]

    ys2 = slice(max(0, y - r_bg), min(frame.shape[0], y + r_bg))
    xs2 = slice(max(0, x - r_bg), min(frame.shape[1], x + r_bg))
    bg_local = frame[ys2, xs2]

    return -1 if np.mean(tail_local) > np.mean(bg_local) else 1


def tailfit_simple(
    vid,
    start_point,
    end_point,
    num_points=30,
    display=True,
    variabledelay=True,
    arcradius = 60,
    speed=4.0,                 # 新增：显示倍速，>1 更快，<1 更慢
    fixed_delay_ms=1,           # 新增：variabledelay=False 时使用的延迟
    disable_display_key=ord('q'),  # 新增：按 q 关闭本视频 display（继续算）
    stop_after_video_key=ord('x'), # 新增：按 x 处理完本视频后停止批处理
    point_color=(0, 255,0),  # 新增：点颜色 BGR
    point_radius=2,  # 新增：点大小（圆半径）
    min_width = 8,
    halfwidth = 22
):
    import cv2
    import numpy as np
    import time
    import scipy
    import scipy.ndimage
    import scipy.spatial.distance

    blursize = 7
    # arcradius = 60
    display_point_color = (0, 0, 255)

    taillength = scipy.spatial.distance.euclidean(start_point, end_point)
    tailpoint_spacing = taillength / num_points

    start_vector = end_point - start_point
    start_vector = start_vector / np.linalg.norm(start_vector)

    frame_fit = np.zeros((num_points, 2))
    fitted_tail = []

    print("Starting tailfit on:  ", vid.filepath)
    print("fps is: ", vid.FPS, " and frame count is: ", len(vid))

    # 本视频内可动态关闭显示，但不影响下一段视频
    display_now = bool(display)
    stop_after_this_video = False

    if display_now:
        cv2.namedWindow("frame_display")
        cv2.moveWindow("frame_display", 0, 0)

    starttime = time.time()

    for framenum, frame in enumerate(vid):
        frame_fit[:] = np.nan # 260203 增加

        if display_now:
            frame_display = frame.copy()

        if frame.ndim == 3:
            frame = frame[..., 1]
        frame = cv2.boxFilter(frame, -1, (blursize, blursize))

        ##


        #添加
        # ---------- 预处理：让暗尾巴从亮背景里“凸显”出来 ----------
        # 1) BLACKHAT：突出暗结构（尾巴）并压掉缓慢变化的亮背景
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 19)) # kernel 越大，越强调大尺度背景差异
        frame = cv2.morphologyEx(frame, cv2.MORPH_BLACKHAT, kernel)

        # 2) 压高光（先做）
        p = np.percentile(frame, 99)
        frame = (frame.astype(np.float32) * (255.0 / max(p, 1))).clip(0, 255).astype(np.uint8)

        # 3) CLAHE（你原来漏了 apply）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame = clahe.apply(frame)
        # === ROI 增强：动态用上一帧尾尖做 end，提高“尾巴 vs 背景”对比 ===
        roi_end = end_point
        if framenum > 0 and len(fitted_tail) > 0 and np.all(np.isfinite(fitted_tail[-1][-1])):
            roi_end = fitted_tail[-1][-1].astype(np.float32)

        frame = roi_enhance_tail(frame, start_point.astype(np.float32), roi_end.astype(np.float32))

        # -------------------------------------------------------

        #if framenum % 5 == 0:
        backgroundsign = getbackgroundsign(frame, start_point)

        guess_vector = start_vector
        current = start_point
        frozen = False
        rej_run = 0

        for count in range(num_points):
            tail_tip = int(0.7 * num_points)  # 后30%算尾尖
            local_min_width = min_width if count < tail_tip else max(3, min_width - 2)
            local_sym_thresh = 0.35 if count < tail_tip else 0.45
            local_lat_thresh = 0.6 if count < tail_tip else 0.75

            if count > 0:
                guess_vector = guess_vector / np.linalg.norm(guess_vector)

            """ 
            260203 修改，删除，替换
            arccenter = np.arctan2(*guess_vector)
            lin = np.linspace(-np.pi * .22 + arccenter, np.pi * .22 + arccenter, 60)

            xs = current[0] + arcradius * np.sin(lin) - guess_vector[0] * (arcradius - tailpoint_spacing)
            ys = current[1] + arcradius * np.cos(lin) - guess_vector[1] * (arcradius - tailpoint_spacing)
            x_indices, y_indices = xs.astype(int), ys.astype(int)

            if max(y_indices) >= frame.shape[0] or min(y_indices) < 0 or max(x_indices) >= frame.shape[1] or min(x_indices) < 0:
                y_indices = np.clip(y_indices, 0, frame.shape[0] - 1)
                x_indices = np.clip(x_indices, 0, frame.shape[1] - 1)
                print("Tail got too close to the edge of the frame, clipping search area!")

            guess_slice = frame[y_indices, x_indices].astype(np.int16)  # 或 np.float32
            guess_slice = (backgroundsign * guess_slice).astype(np.int16)

            #normcumslice = np.cumsum(guess_slice - guess_slice[:3].mean() * .5 - guess_slice[-3:].mean() * .5)
            #result_index = (np.abs(normcumslice - normcumslice[-1] * .5)).argmin()
            #newpoint = np.array([x_indices[result_index], y_indices[result_index]])
            #修改为
            # 预测：下一点应该大概在 current + guess_vector * tailpoint_spacing 附近
            pred = current + guess_vector * tailpoint_spacing

            normcumslice = np.cumsum(
                guess_slice - guess_slice[:3].mean() * .5 - guess_slice[-3:].mean() * .5
            )
            mid = normcumslice[-1] * 0.5

            # 原始强度目标：离 mid 越近越好
            score_intensity = np.abs(normcumslice - mid)

            # 连续性惩罚：离预测位置越远越差（防止跳到背景杂质）
            dxp = x_indices - pred[0]
            dyp = y_indices - pred[1]
            score_dist = np.sqrt(dxp * dxp + dyp * dyp)

            # 组合评分：lambda 越大越“黏”预测位置（推荐 0.8 ~ 2.0）
            lam = 1
            score = score_intensity + lam * score_dist

            result_index = np.argmin(score)
            newpoint = np.array([x_indices[result_index], y_indices[result_index]])
            """

            # ===== 新：横截面搜索（在预测点附近，沿法线找中心线）=====

            # 1) 预测下一个点的位置（沿当前方向走一个 spacing）
            gv= guess_vector.astype(np.float32)
            gv = gv / (np.linalg.norm(gv) + 1e-9)

            # --- 修正版 Step 1：pred 以本帧方向 gv 为主 + 上一帧同点位移小修正（限幅），避免“缩成一团” ---
            pred = current.astype(np.float32) + gv * float(tailpoint_spacing)
            gv_used = gv  # 保持你的后续代码不动（n、frozen 时用 gv_used）

            if framenum > 0 and len(fitted_tail) > 0:
                prev_pt = fitted_tail[-1][count].astype(np.float32)

                # 保护：上一帧该点可能是 nan
                if np.all(np.isfinite(prev_pt)):
                    delta = prev_pt - current.astype(np.float32)

                    # 限制修正幅度，防止异常把 pred 拉回去/拉飞
                    max_step = 0.8 * float(tailpoint_spacing)
                    dnorm = np.linalg.norm(delta)
                    if dnorm > max_step:
                        delta = delta * (max_step / (dnorm + 1e-9))

                    # 只加小权重（0.15~0.30），不会主导 pred
                    pred = pred + 0.15 * delta

            # 2) 法线方向（垂直于尾巴方向）
            n = np.array([-gv_used[1], gv_used[0]], dtype=np.float32)

            # 3) 横截面半宽：建议先用 18~28 之间。你 spacing=22，我建议先用 22，管 搜索范围
            hw = halfwidth if rej_run < 4 else int(halfwidth * 1.8)  # 连续丢失就扩大视野
            t = np.linspace(-hw, hw, 2 * hw + 1)

            xs = pred[0] + t * n[0]
            ys = pred[1] + t * n[1]
            x_indices = np.clip(xs.astype(int), 0, frame.shape[1] - 1)
            y_indices = np.clip(ys.astype(int), 0, frame.shape[0] - 1)

            # 4) 取 profile，并修 dtype（避免 uint8 * -1）
            guess_slice = frame[y_indices, x_indices].astype(np.int16)
            guess_slice = (backgroundsign * guess_slice).astype(np.int16)

            # 5) 平滑一下 profile
            gs = guess_slice.astype(np.float32)
            if len(gs) >= 7:
                gs = cv2.GaussianBlur(gs.reshape(-1, 1), (1, 7), 0).ravel()

            # 6) 用梯度找两条边缘，再取中点（更稳地落在“中心线”）
            dg = np.diff(gs)
            i_pos = int(np.argmax(dg))
            i_neg = int(np.argmin(dg))
            edge1, edge2 = i_pos, i_neg
            if edge1 > edge2:
                edge1, edge2 = edge2, edge1

            # ===== 补丁1：左右对称性检查（防背景污渍）=====
            center0 = hw
            cand = int(0.5 * (edge1 + edge2))

            w = 3
            l0 = max(0, cand - w)
            l1 = cand
            r0 = cand + 1
            r1 = min(len(gs), cand + w + 1)

            left_mean = np.mean(gs[l0:l1]) if l1 > l0 else 0
            right_mean = np.mean(gs[r0:r1]) if r1 > r0 else 0

            sym = abs(left_mean - right_mean) / (abs(left_mean) + abs(right_mean) + 1e-6)

            if sym > local_sym_thresh:
                rejected = True
                if framenum > 0:
                    alpha = 0.6 if count < tail_tip else 0.35  # 尾尖更信上一帧
                    newpoint = alpha * pred + (1 - alpha) * fitted_tail[-1][count].astype(np.float32)
                else:
                    newpoint = pred

            else:
                newpoint = np.array([x_indices[cand], y_indices[cand]], dtype=np.float32)
                rejected = False
            # ===============================================
            # ===== 补丁2：横向位移限制（防横跳到污渍）=====
            lateral = abs(np.dot(newpoint - pred, n))
            if lateral > local_lat_thresh * hw :
                rejected = True
                if framenum > 0:
                    # 用上一帧同一点作为强先验，避免尾尖变直或飞走
                    newpoint = 0.6 * pred + 0.4 * fitted_tail[-1][count].astype(np.float32)
                else:
                    newpoint = pred

            # ===============================================

            # 7) 宽度门槛：太窄就认为没切到尾巴（回退 pred，并标记 rejected）
            width = (edge2 - edge1)
            if width < local_min_width:
                rejected = True
                if framenum > 0:
                    # 用上一帧同一点作为强先验，避免尾尖变直或飞走
                    newpoint = 0.6 * pred + 0.4 * fitted_tail[-1][count].astype(np.float32)
                else:
                    newpoint = pred


            # =========================================

            #————————————————————————————————————————————————————
            # 连续 rejected 计数
            if rejected:
                rej_run += 1
            else:
                rej_run = 0

            # 连续丢失太久 → 解冻，准备重新抓回
            if rej_run >= 4:
                frozen = False
            # ————————————————————————————————————————————————————

            if display_now:

                col_pt = (0, 0, 255) if rejected else (0, 255, 0)  # rejected=红，正常=绿
                cv2.circle(frame_display, (int(newpoint[0]), int(newpoint[1])), int(point_radius), col_pt, -1)
                #cv2.circle(frame_display, (int(newpoint[0]), int(newpoint[1])), int(point_radius), point_color)

                # =====（可选）在 display 上把横截面画出来，方便你看它扫到哪 =====
                if rejected:
                    # 被拒绝 / 回退 → 红色
                    frame_display[y_indices, x_indices] = (0, 0,255 )
                else:
                    # 正常切到尾巴 → 绿色
                    frame_display[y_indices, x_indices] = (0,255,0)

            frame_fit[count, :] = newpoint

            if count > 0:
                if rejected and rej_run >= 2:
                    # 冻结方向：后面都不再更新方向
                    frozen = True
                    if rejected and display_now and (count == int(0.9 * num_points)):
                        print(
                            f"[frame {framenum}] rejected at count={count}, width={width}, sym={sym:.2f}, lateral={lateral:.1f}, rej_run={rej_run}")

                if not frozen:
                    guess_vector = (newpoint - current).astype(np.float32)
                    guess_vector = guess_vector / (np.linalg.norm(guess_vector) + 1e-9)
                else:
                    # frozen 时保持原方向（用上一轮的 gv）
                    guess_vector = gv_used

            current = newpoint

        fitted_tail.append(np.copy(frame_fit))

        # -------- 显示与按键处理 --------
        if display_now:
           
            cv2.putText(frame_display, str(framenum), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,255))
            cv2.imshow("frame_display", frame_display)

            if framenum == 0:
                delaytime = 1
            else:
                if variabledelay:
                    minlen = min([fitted_tail[-2].shape[0], fitted_tail[-1].shape[0]]) - 1
                    delaytime = int(min(max((np.abs(((fitted_tail[-2][minlen, :] - fitted_tail[-1][minlen, :]) ** 2).sum() ** .5) ** 1.2 * 3 - 1), 1), 500))
                else:
                    delaytime = int(fixed_delay_ms)

            # 倍速：speed 越大越快（delay 越小）
            delaytime = max(1, int(delaytime / max(speed, 1e-6)))

            key = cv2.waitKey(delaytime) & 0xFF

            # q：本视频关闭显示，但继续计算
            if key == disable_display_key:
                display_now = False
                try:
                    cv2.destroyWindow("frame_display")
                except Exception:
                    pass

            # x：处理完本视频后停止批处理
            if key == stop_after_video_key:
                stop_after_this_video = True

            # ESC：立刻抛 KeyboardInterrupt（主程序会写 checkpoint）
            if key == 27:
                raise KeyboardInterrupt

    print("Tailfit done in %.2f seconds" % (time.time() - starttime))
    # 返回 stop 标记，主程序决定是否继续下一个文件
    return fitted_tail, stop_after_this_video

# ——————————————————————————————————————————————————————————————————————————————

# 新代码
import numpy as np
import cv2
import time
import scipy.spatial.distance
import time

def skeletonize_fast(bw255):
    # 只打印一次：当前用哪个 backend
    if not getattr(skeletonize_fast, "_printed", False):
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            backend = "opencv-ximgproc.thinning"
        else:
            backend = "skimage.skeletonize or zhang_suen (fallback)"
        print(">>> skeletonize backend:", backend)
        skeletonize_fast._printed = True
    # 1) 优先用 OpenCV contrib（最快）
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(bw255)

    # 2) 其次用 skimage（也比纯 python 快不少）
    try:
        from skimage.morphology import skeletonize
        return (skeletonize(bw255 > 0).astype(np.uint8) * 255)
    except Exception:
        pass

    # 3) 最后 fallback：你的 Zhang–Suen
    return zhang_suen_thinning(bw255)




# ----------------------------
# 1) ROI mask：用上一帧 polyline（或第一帧 start-end 线）画粗线当 ROI
# ----------------------------
def make_roi_mask(shape_hw, polyline_xy, thickness=60, dilate=9):
    h, w = shape_hw
    mask = np.zeros((h, w), np.uint8)

    pts = np.asarray(polyline_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=int(thickness))

    if dilate and dilate > 0:
        k = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


# ----------------------------
# 2) 在 ROI 内把尾巴分割成二值 mask（暗尾巴：THRESH_BINARY_INV）
# ----------------------------
def tail_mask_from_roi(frame_gray, roi_mask):
    g = frame_gray.copy()
    g = cv2.GaussianBlur(g, (0, 0), 1.2)

    tmp = g.copy()
    tmp[roi_mask == 0] = 255  # ROI 外置白，避免阈值被背景影响

    # Otsu 阈值（暗尾巴）
    _, bw = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学：补断裂 + 去小噪点
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8), iterations=1)

    bw[roi_mask == 0] = 0
    return bw


# ----------------------------
# 3) 纯 numpy 的 Zhang-Suen thinning（不依赖 skimage / ximgproc）
#    输出 skeleton (uint8 0/255)
# ----------------------------
def zhang_suen_thinning(bw255):
    img = (bw255 > 0).astype(np.uint8)
    changed = True
    h, w = img.shape

    def neighbors(x, y):
        # P2..P9 (clockwise)
        p2 = img[x-1, y]
        p3 = img[x-1, y+1]
        p4 = img[x,   y+1]
        p5 = img[x+1, y+1]
        p6 = img[x+1, y]
        p7 = img[x+1, y-1]
        p8 = img[x,   y-1]
        p9 = img[x-1, y-1]
        return [p2, p3, p4, p5, p6, p7, p8, p9]

    def transitions(ps):
        # number of 0->1 transitions in ordered neighbors
        s = 0
        for i in range(8):
            if ps[i] == 0 and ps[(i+1) % 8] == 1:
                s += 1
        return s

    it = 0
    while changed and it < 200:  # 上限防死循环
        changed = False
        it += 1

        # Step 1
        to_remove = []
        for x in range(1, h-1):
            for y in range(1, w-1):
                if img[x, y] != 1:
                    continue
                ps = neighbors(x, y)
                B = sum(ps)
                A = transitions(ps)
                if 2 <= B <= 6 and A == 1 and ps[0]*ps[2]*ps[4] == 0 and ps[2]*ps[4]*ps[6] == 0:
                    to_remove.append((x, y))
        if to_remove:
            for x, y in to_remove:
                img[x, y] = 0
            changed = True

        # Step 2
        to_remove = []
        for x in range(1, h-1):
            for y in range(1, w-1):
                if img[x, y] != 1:
                    continue
                ps = neighbors(x, y)
                B = sum(ps)
                A = transitions(ps)
                if 2 <= B <= 6 and A == 1 and ps[0]*ps[2]*ps[6] == 0 and ps[0]*ps[4]*ps[6] == 0:
                    to_remove.append((x, y))
        if to_remove:
            for x, y in to_remove:
                img[x, y] = 0
            changed = True

    return (img * 255).astype(np.uint8)


# ----------------------------
# 4) skeleton 上找“从 start 走到最远端”的路径（8邻域 BFS）
#    返回 polyline_xy: Nx2 (x,y)
# ----------------------------
def skeleton_longest_path_from_start(sk255, start_xy,dir_xy=None):
    sk = (sk255 > 0)
    ys, xs = np.where(sk)
    if len(xs) < 20:
        return None,None

    # 找 skeleton 上离 start 最近的像素作为起点
    sx, sy = float(start_xy[0]), float(start_xy[1])
    d2 = (xs - sx)**2 + (ys - sy)**2
    i0 = int(np.argmin(d2))
    start_pix = (int(xs[i0]), int(ys[i0]))  # (x,y)

    h, w = sk.shape
   
    # ✅ 方向单位向量：只在这里算一次
    dv = None
    if dir_xy is not None:
        dv = np.asarray(dir_xy, np.float32)
        dn = float(np.hypot(dv[0], dv[1]))
        if dn > 1e-6:
            dv /= dn
        else:
            dv = None  # 方向太小就当没提供
            
    # BFS
    from collections import deque
    dist = -np.ones((h, w), np.int32)
    prev = -np.ones((h, w, 2), np.int32)

    q = deque()
    q.append((start_pix[1], start_pix[0]))  # store as (y,x)
    dist[start_pix[1], start_pix[0]] = 0

    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    far_yx = (start_pix[1], start_pix[0])
    far_d = 0

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        if dir_xy is None:
            # 没提供方向：保持原逻辑（最远距离）
            if d > far_d:
                far_d = d
                far_yx = (y, x)
        else:
            # 提供方向：只允许往尾尖方向（投影>0）的点作为候选终点
            vx = x - start_pix[0]
            vy = y - start_pix[1]
            proj = vx * dv[0] + vy * dv[1]
            if proj > 0 and d > far_d:
                far_d = d
                far_yx = (y, x)


        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not sk[ny, nx]:
                continue
            if dist[ny, nx] >= 0:
                continue
            dist[ny, nx] = d + 1
            prev[ny, nx, :] = (y, x)
            q.append((ny, nx))

    # 回溯得到路径
    path = []
    y, x = far_yx
    if dist[y, x] < 5:
        return None,None

    while True:
        path.append((x, y))
        py, px = prev[y, x]
        if py < 0:
            break
        y, x = int(py), int(px)

    path.reverse()
    poly = np.asarray(path, dtype=np.float32)
    return poly, start_pix
# ----------------------------
# 5) 将 polyline 按弧长重采样成 num_points 个点
# ----------------------------
def resample_polyline(poly_xy, num_points):
    pts = np.asarray(poly_xy, np.float32)
    if len(pts) < 2:
        return None,None

    seg = np.diff(pts, axis=0)
    seglen = np.sqrt((seg**2).sum(axis=1))
    L = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(L[-1])
    if total < 1e-6:
        return None

    target = np.linspace(0, total, num_points)
    out = np.zeros((num_points, 2), np.float32)

    j = 0
    for i, t in enumerate(target):
        while j < len(L) - 2 and L[j+1] < t:
            j += 1
        t0, t1 = L[j], L[j+1]
        p0, p1 = pts[j], pts[j+1]
        if t1 - t0 < 1e-9:
            out[i] = p0
        else:
            a = (t - t0) / (t1 - t0)
            out[i] = (1 - a) * p0 + a * p1
    return out


# ----------------------------
# 6) 主函数：tailfit_skeleton（用它替代你原来的 tailfit_simple）
# ----------------------------


def tailfit_skeleton(

    vid,
    start_point,
    end_point,
    num_points=30,
    display=True,
    speed=4.0,
    roi_thickness=120,     # ROI 粗细：越大越不容易丢，但越容易带进杂质（60~90常用）
    roi_dilate=15,
    min_mask_area=500,   # mask 太小就认为分割失败（按你图像大小调，先用 2000）
    point_radius=2,
    endpoint_y_margin=10,  # 允许超出尾尖的像素（5~20 之间调）
    enable_endpoint_gate=True,
    
    enable_temporal_gate=True, # 开启则按每帧计算方向
    temporal_angle_deg_max=60.0,   # 方向突变阈值（度）
    temporal_rms_px_max=20.0,      # 形状突变阈值（像素）
    temporal_use_points=(0, 3),    # 用第0和第3点估计“尾根方向”

):
    start_point = np.asarray(start_point, np.float32)
    end_point   = np.asarray(end_point,   np.float32)
    # ===== y fixed + x updated anchor =====
    y_anchor = float(start_point[1])   # 固定 y
    x_anchor = float(start_point[0])   # 初始 x
    # =====================================



    print("Starting tailfit_skeleton on:", vid.filepath)
    print("fps:", vid.FPS, "frames:", len(vid))

    fitted_tail = []
    stop_after_this_video = False

    display_now = bool(display)
    if display_now:
        cv2.namedWindow("frame_display")
        cv2.moveWindow("frame_display", 0, 0)

    # 初始 polyline：用 start-end 直线
    init_line = np.linspace(start_point, end_point, num_points).astype(np.float32)

    t0 = time.time()

    # ====== 方案2：固定尾根前 K 个点为直线 ======
    K_BASE = 3  # 建议 3~5，先用 4
    
    base_dir0 = (end_point - start_point).astype(np.float32)
    base_dir0 /= (np.linalg.norm(base_dir0) + 1e-6)
    
    spacing0 = float(np.linalg.norm(init_line[1] - init_line[0]))
    
    def enforce_base_straight(fit_xy, start_xy):
        if fit_xy is None:
            return fit_xy
    
        fit_xy = fit_xy.copy()
    
        for i in range(min(K_BASE, fit_xy.shape[0])):
            fit_xy[i] = start_xy + base_dir0 * (i * spacing0)
    
        return fit_xy
    # ==============================================
    
    # ================= QC counters =================
    qc_fallback_frames = 0  # 分割失败 → 复制上一帧
    qc_short_skel_frames = 0  # skeleton 太短
    qc_consecutive_fallback = 0
    qc_max_consecutive_fallback = 0
    qc_endpoint_gate = 0
    qc_poly_none = 0
    qc_poly_too_short = 0
    qc_straight_gate = 0
    qc_other_gate = 0
    qc_temporal_gate = 0


    # ===============================================

    for framenum, frame in enumerate(vid):
        t_frame0 = time.perf_counter()
        t_bw = t_sk = t_bfs = t_fit = t_show = None

        if frame.ndim == 3:
            frame_gray = frame[..., 1].copy()
        else:
            frame_gray = frame.copy()

        if display_now:
            frame_display = frame.copy()
       
        # ===== y fixed + x updated =====
        # 用上一帧拟合结果的第一个红点更新 x（最稳、你现在就能用）
        if framenum > 0 and len(fitted_tail) > 0 and np.all(np.isfinite(fitted_tail[-1])):
            x_anchor = float(fitted_tail[-1][0, 0])   # 上一帧红点第1个点的 x
        # 限制 x 不要跑出画面
        H, W = frame_gray.shape[:2]
        x_anchor = float(np.clip(x_anchor, 0, W - 1))
        
        start_point = np.array([x_anchor, y_anchor], np.float32)
        # =================================
        

        # 1) 构造 ROI：优先用上一帧拟合出来的 polyline
        if framenum > 0 and len(fitted_tail) > 0 and np.all(np.isfinite(fitted_tail[-1])):
            poly_for_roi = fitted_tail[-1]
        else:
            poly_for_roi = init_line

        roi_mask = make_roi_mask(frame_gray.shape[:2], poly_for_roi, thickness=roi_thickness, dilate=roi_dilate)
        
        # ⭐ 方案一：强制 ROI 包含 start_point 附近
        cv2.circle(
            roi_mask,
            (int(start_point[0]), int(start_point[1])),
            25,     # 半径，建议先用 25
            255,
            -1
        )

        # 2) 分割尾巴 mask
        # === 已经是二值图：尾巴黑(0)，背景白(255) ===
        bw = np.zeros_like(frame_gray, dtype=np.uint8)
        bw[frame_gray < 128] = 255  # 黑尾巴->255 (前景)

        # 只保留 ROI 内（防止其它黑点）
        bw[roi_mask == 0] = 0

        # ===== A: head cut by start_point =====
        head_cut_margin = 5  # 允许保留 start_point 上方的余量(像素)，0~15 常用
        
        ycut = int(start_point[1]) - head_cut_margin
        if ycut > 0:
            bw[:ycut, :] = 0
        # =====================================
        
        t_bw = time.perf_counter()

        # ===== 新增：只保留包含 start_point 的连通域 =====
        num, labels, stats, _ = cv2.connectedComponentsWithStats((bw > 0).astype(np.uint8), 8)
        
        if num > 1:
            sx, sy = int(start_point[0]), int(start_point[1])
            if 0 <= sy < labels.shape[0] and 0 <= sx < labels.shape[1]:
                lab = labels[sy, sx]
                if lab != 0:  # 0 是背景
                    bw = np.where(labels == lab, 255, 0).astype(np.uint8)
        # =================================================



        # 若分割太失败，就 fallback：用上一帧点（或初始线）
        if int(np.sum(bw > 0)) < int(min_mask_area):
            qc_fallback_frames += 1
            qc_consecutive_fallback += 1
            qc_max_consecutive_fallback = max(
                qc_max_consecutive_fallback, qc_consecutive_fallback
            )

            if framenum > 0 and len(fitted_tail) > 0:
                fit = fitted_tail[-1].copy()
            else:
                fit = init_line.copy()
          
            fit = enforce_base_straight(fit, start_point)
            fitted_tail.append(fit)
        
            # 6) display（只显示每 show_every 帧，避免卡顿）
            show_every = 5  # ⭐ 你可以改成 3 / 10

            if display_now and (framenum % show_every == 0):
                fit = fitted_tail[-1]

                for p in fit:
                    cv2.circle(frame_display, (int(p[0]), int(p[1])),
                               int(point_radius), (0, 0, 255), -1)

                cv2.putText(frame_display, str(framenum), (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                cv2.imshow("frame_display", frame_display)

                # ⭐ 这里一定要用 waitKey(1)，不要再用 delay
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt

            continue


        # 3) skeletonize（# 用 ROI 决定 crop（bbox 不再被噪点撑大）
        ys, xs = np.where(roi_mask  > 0)
        pad = 10  # bbox 外扩像素，防止切掉尾巴边缘
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(bw.shape[0], int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(bw.shape[1], int(xs.max()) + pad + 1)
        
        # ===== 新增：打印 bbox 大小 =====
        if framenum % 30 == 0:
            print(f"[DBG] bbox size = {x1-x0} x {y1-y0}")
        # =================================

        bw_crop = bw[y0:y1, x0:x1]
        # crop 内也严格只保留 ROI 内的前景
        roi_crop = roi_mask[y0:y1, x0:x1]
        bw_crop[roi_crop == 0] = 0

        sk_crop = skeletonize_fast(bw_crop)
        t_sk = time.perf_counter()

        # ===== A: 清边界 skeleton，禁止 BFS 走到 crop 边框 =====
        border = 2   # 1~3；你这种 30 帧连续出界，先用 2
        sk_crop[:border, :] = 0
        sk_crop[-border:, :] = 0
        sk_crop[:, :border] = 0
        sk_crop[:, -border:] = 0
        t_sk = time.perf_counter()

        # ======================================================

        # 4) 从 start 出发找最长路径 polyline（注意 start_point 要映射到 crop 坐标）
        sp_crop = start_point.copy()
        sp_crop[0] -= x0
        sp_crop[1] -= y0

        # dir 向量：用 (end_point - start_point)，然后转到 crop 坐标（其实向量不需要减 x0/y0）
        dir_xy = (end_point - start_point).copy()
        
        poly, start_pix = skeleton_longest_path_from_start(sk_crop, sp_crop, dir_xy=dir_xy)
        t_bfs = time.perf_counter()


        use_fallback = False
        poly_full = None  # 先定义，防止后面引用未定义

        # ---- 规则0：poly 太短 / 没有 ----
        if poly is None:
            qc_poly_none += 1
            qc_short_skel_frames += 1
            use_fallback = True
        elif len(poly) < 10:
            qc_poly_too_short += 1
            qc_short_skel_frames += 1
            use_fallback = True
        else:
            # 1) 先把 poly 加回全图坐标
            poly_full = poly.copy()
            poly_full[:, 0] += x0
            poly_full[:, 1] += y0

            # ---- 规则1：endpoint（尾尖最低点）门控：终点不许跑到尾尖“下面” ----
            if enable_endpoint_gate:
                end_y = float(end_point[1])
                poly_end_y = float(poly_full[-1, 1])  # skeleton 找到的最远端 y

            # 把 poly_full 裁到 y 不超过阈值的位置
            if poly_end_y > end_y + endpoint_y_margin:
                qc_endpoint_gate += 1
            
                ylimit = end_y + float(endpoint_y_margin)
            
                # 找到 poly_full 中最后一个 y <= ylimit 的点
                ok = np.where(poly_full[:, 1] <= ylimit)[0]
                if len(ok) >= 10:
                    poly_full = poly_full[: ok[-1] + 1]   # ✅ 截断，而不是 fallback
                else:
                    use_fallback = True
                    qc_short_skel_frames += 1
            

            # ---- 规则2：edge gate（四边兜底）----
            H, W = frame_gray.shape[:2]
            edge_margin = 6  # 先用 6；如果你想更宽容可以改 4
            
            endx = float(poly_full[-1, 0])
            endy = float(poly_full[-1, 1])
            
            if (endx < edge_margin or endx > (W - 1 - edge_margin) or
                endy < edge_margin or endy > (H - 1 - edge_margin)):
                use_fallback = True
                qc_other_gate += 1
                qc_short_skel_frames += 1
            # ---------------------------------



        # ---- 执行 fallback 或正常输出 ----
        if use_fallback:
            # （可选但建议）统计 fallback
            qc_fallback_frames += 1
            qc_consecutive_fallback += 1
            qc_max_consecutive_fallback = max(qc_max_consecutive_fallback, qc_consecutive_fallback)
            
            if framenum > 0 and len(fitted_tail) > 0:
                fit = fitted_tail[-1].copy()
            else:
                fit = init_line.copy()

            fit = enforce_base_straight(fit, start_point)
   
            fitted_tail.append(fit)
            t_fit = time.perf_counter()

        else:
            # 这里用 poly_full（全图坐标）重采样
            fit = resample_polyline(poly_full, num_points)
        
            # 如果重采样失败：fallback
            if fit is None:
                qc_fallback_frames += 1
                qc_consecutive_fallback += 1
                qc_max_consecutive_fallback = max(qc_max_consecutive_fallback, qc_consecutive_fallback)
        
                if framenum > 0 and len(fitted_tail) > 0:
                    fit = fitted_tail[-1].copy()
                else:
                    fit = init_line.copy()
                
                fit = enforce_base_straight(fit, start_point)

                fitted_tail.append(fit)
                continue
        
            # ===== Temporal continuity gate（用 start->tailtip 方向，防止尾尖跳变）=====
            if enable_temporal_gate and framenum > 0 and len(fitted_tail) > 0:
                prev = fitted_tail[-1]
            
                # 用 start_point 到尾尖(最后一点) 的方向向量
                v_prev = prev[-1] - start_point
                v_cur  = fit[-1]  - start_point
            
                n_prev = float(np.hypot(v_prev[0], v_prev[1])) + 1e-6
                n_cur  = float(np.hypot(v_cur[0],  v_cur[1])) + 1e-6
                v_prev /= n_prev
                v_cur  /= n_cur
            
                cosang = float(np.clip(v_prev[0]*v_cur[0] + v_prev[1]*v_cur[1], -1.0, 1.0))
                ang_deg = float(np.degrees(np.arccos(cosang)))
            
                # 如果尾尖方向突然变化太大：认为这帧尾尖跳了
                if ang_deg > temporal_angle_deg_max:
                    qc_temporal_gate += 1
            
                    # ✅ 软纠正：不要直接冻结(prev)，避免锁死
                    alpha = 0.2  # 0.1~0.3；越小越保守
                    fit = (1 - alpha) * prev + alpha * fit
            
                    fitted_tail.append(fit)
                    qc_consecutive_fallback = 0
                    continue
            # ===== Temporal gate end =====

        
            # 通过 temporal gate：正常输出
            fitted_tail.append(fit)
            qc_consecutive_fallback = 0


        # （可选）如果你想 debug 显示 skeleton，可把 crop skeleton 放回整图
        # sk = np.zeros_like(bw)
        # sk[y0:y1, x0:x1] = sk_crop

        # 6) display
        if display_now:
            fit = fitted_tail[-1]

            # 画 ROI 边界（可选）
            # frame_display[roi_mask > 0] = frame_display[roi_mask > 0]  # 不改色，留着
            # 画点（红）
            for p in fit:
                cv2.circle(frame_display, (int(p[0]), int(p[1])), int(point_radius), (0, 0, 255), -1)

            # 蓝点：你传入的 start_point（理论尾根，固定）
            cv2.circle(
                frame_display,
                (int(start_point[0]), int(start_point[1])),
                5,
                (255, 0, 0),
                -1
            )
            # 黄点：算法投影到 skeleton 上的真实起点
            if start_pix is not None:
                sx = int(start_pix[0] + x0)
                sy = int(start_pix[1] + y0)
                cv2.circle(
                    frame_display,
                    (sx, sy),
                    5,
                    (0, 255, 255),
                    -1
                )
                    
            # ⭐⭐⭐ 画绿色 cut 线（一定放在 imshow 之前）
            H, W = frame_display.shape[:2]
            yline = int(np.clip(ycut, 0, H-1))
            cv2.line(frame_display,
                     (0, yline),
                     (W-1, yline),
                     (0, 255, 0), 2)
        
            # 打印 ycut 值
            cv2.putText(frame_display, f"ycut={ycut}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0), 2)

            cv2.putText(frame_display, str(framenum), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            cv2.imshow("frame_display", frame_display)

            delay = max(1, int(10 / max(speed, 1e-6)))
            key = cv2.waitKey(delay) & 0xFF
            t_show = time.perf_counter()

            if key == 27:
                raise KeyboardInterrupt
            t_show = time.perf_counter()

            if framenum % 30 == 0 and None not in (t_bw, t_sk, t_bfs, t_fit, t_show):
                print(
                    f"[TIMING] frame {framenum:04d} | "
                    f"bw={(t_bw - t_frame0)*1000:6.1f}ms, "
                    f"sk={(t_sk - t_bw)*1000:6.1f}ms, "
                    f"bfs={(t_bfs - t_sk)*1000:6.1f}ms, "
                    f"fit={(t_fit - t_bfs)*1000:6.1f}ms, "
                    f"show={(t_show - t_fit)*1000:6.1f}ms"
                )


    print("tailfit_skeleton done in %.2f s" % (time.time() - t0))
    n_frames = len(fitted_tail)
    fallback_ratio = 100.0 * qc_fallback_frames / max(n_frames, 1)
    short_ratio    = 100.0 * qc_short_skel_frames / max(n_frames, 1)
   

    print(
            f"————————————————————————————————————————————————————\n  "
            f"[QC skeleton] {vid.filepath}\n"
            f"  {'fallback':<28}: {qc_fallback_frames:>4}/{n_frames:<4} ({fallback_ratio:>5.1f}%)  "
            f"- frame replaced by previous fit\n"
            f"  {'short_skel':<28}: {qc_short_skel_frames:>4}/{n_frames:<4} ({short_ratio:>5.1f}%)  "
            f"- skeleton too short or unreliable\n"
            f"  {'max_consecutive_fallback':<28}: {qc_max_consecutive_fallback:>4}  "
            f"- longest frozen sequence\n"
            f"————————————————————————————————————————————————————\n"
        )
    #  | 指标              | 经验安全线   |
    # | --------------- | ------- |
    # | fallback / 总帧   | < 5%    |  在 B 帧里，有 A 帧 算法已经放弃追踪，直接用了“上一帧的尾巴点”（冻结帧）。
    # | short_skel / 总帧 | < 10%   |  在 B 帧里，有 C 帧 虽然没完全失败，但“可见尾巴太短”，
    # | max_consecutive | < 15–20 | 最长的一段连续“不可完全信任帧”有 D 帧。
    print(
        f"[QC detail] {vid.filepath}\n"
        f"  {'poly_none':<28}: {qc_poly_none:>4}  "
        f"- no valid skeleton path found\n"
        f"  {'poly_short':<28}: {qc_poly_too_short:>4}  "
        f"- detected path too short\n"
        f"  {'endpoint_gate':<28}: {qc_endpoint_gate:>4}  "
        f"- endpoint exceeded tail-tip constraint\n"
        f"  {'straight_gate':<28}: {qc_straight_gate:>4}  "
        f"- abnormal straight tail detected\n"
        f"  {'other_gate':<28}: {qc_other_gate:>4}  "
        f"- endpoint near image boundary\n"
        f"  {'temporal_gate':<28}: {qc_temporal_gate:>4}  "
        f"- sudden frame-to-frame shape change\n"
    
    )
    
    # ======= Auto-save BAD QC (super simple) =======
    try:

        is_bad = (
            qc_fallback_frames >= 30 or
            qc_short_skel_frames >= 30 or
            qc_max_consecutive_fallback >= 30 or
            qc_endpoint_gate >= 30
        )
        if is_bad:
            output_folder = os.path.dirname(vid.filepath)
            bad_path = os.path.join(output_folder, "bad_qc.txt")
    
            print(">>> TRY WRITE bad_qc.txt:", bad_path)  # 🔴 关键调试行
    
            with open(bad_path, "a", encoding="utf-8") as f:
                f.write(f"FILE: {vid.filepath}\n")
                f.write(
                    f"[QC skeleton] {vid.filepath} | "
                    f"fallback={qc_fallback_frames}/{n_frames}, "
                    f"short_skel={qc_short_skel_frames}/{n_frames}, "
                    f"max_consecutive_fallback={qc_max_consecutive_fallback}\n"
                )
                f.write(
                    f"[QC detail] {vid.filepath} | "
                    f"poly_none={qc_poly_none}, "
                    f"poly_short={qc_poly_too_short}, "
                    f"endpoint_gate={qc_endpoint_gate}, "
                    f"straight_gate={qc_straight_gate}, "
                    f"other_gate={qc_other_gate}\n"
                )
                f.write("-" * 80 + "\n")
    
            print(">>> WRITE DONE")  # 🔴 关键调试行
    
    except Exception as e:
        print("!!! QC WRITE FAILED !!!", e)  # 🔴 不要 pass
    # ==============================================


    return fitted_tail, stop_after_this_video



'''
用上一帧尾巴点画一个“允许搜索区域”（ROI）

在 ROI 里把尾巴二值化成前景（bw）

对 bw 做骨架化（skeleton）

从尾根附近出发，在骨架上找“最长那条路径”

把这条路径等距重采样成固定数量的点（红点）

做各种门控（gate）和 fallback，保证坏帧不污染后续
'''
