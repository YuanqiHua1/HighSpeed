import os, shutil, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

PKL_FOLDER  = r"U:\YuanqiHua\High speed\260123 wt1 MTZ\pkl_output"
KEEP_FOLDER = r"U:\YuanqiHua\High speed\260123 wt1 MTZ\pkl_keep"

DEFAULT_FRAMES = [0, 50, 100, 200]
EPS = 1e-3

def load_pkl(p):
    with open(p, "rb") as f:
        tf = pickle.load(f)
    return np.asarray(tf, dtype=float)

def parse_frames(text, n_frames):
    try:
        frames = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
        frames = [max(0, min(n_frames-1, fr)) for fr in frames]
        frames = sorted(list(dict.fromkeys(frames)))
        return frames if frames else None
    except:
        return None

def draw_main(fig_main, tf, frames_to_plot, title):
    fig_main.clf()
    ax1 = fig_main.add_subplot(2, 2, 1)
    ax2 = fig_main.add_subplot(2, 2, 3)
    ax3 = fig_main.add_subplot(2, 2, 4)

    # tail shape
    for fr in frames_to_plot:
        pts = tf[fr]
        ax1.plot(pts[:,0], pts[:,1], "-o", ms=3, lw=1, label=f"{fr}")
    ax1.invert_yaxis()
    ax1.axis("equal")
    ax1.set_title("Tail shape (frames)")
    ax1.legend(title="frame", fontsize=8)

    # tailtip x
    tip = tf[:, -1, :]
    x = tip[:,0]
    ax2.plot(x, lw=1)
    ax2.set_title("Tail tip x over time")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("x")

    # displacement
    step = np.linalg.norm(np.diff(tip, axis=0), axis=1)
    ax3.plot(step, lw=1)
    ax3.axhline(EPS, ls="--")
    ax3.set_yscale("log")
    ax3.set_title("Tail tip displacement (log)")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("step")

    # info text
    frac0 = float(np.mean(step < EPS)) if step.size else np.nan
    endwin = min(60, step.size) if step.size else 0
    frac0_end = float(np.mean(step[-endwin:] < EPS)) if endwin else np.nan
    ax1.text(0.02, 0.02, f"{title}\nfrac(step<EPS)={frac0:.3f}\nlast{endwin}={frac0_end:.3f}",
             transform=ax1.transAxes, fontsize=9, va="bottom")

    fig_main.tight_layout()
    fig_main.canvas.draw_idle()

def review_folder():
    os.makedirs(KEEP_FOLDER, exist_ok=True)
    pkls = sorted([f for f in os.listdir(PKL_FOLDER) if f.lower().endswith(".pkl")])

    for fn in pkls:
        path = os.path.join(PKL_FOLDER, fn)
        tf = load_pkl(path)
        n_frames = tf.shape[0]
        frames_to_plot = [fr for fr in DEFAULT_FRAMES if fr < n_frames] or [0]

        print("\n==============================", flush=True)
        print("FILE:", fn, flush=True)
        print("Frames:", n_frames, flush=True)
        print("Default frames_to_plot:", frames_to_plot, flush=True)
        print("==============================\n", flush=True)

        decision = {"val": None}

        # ---- main figure (plots only) ----
        fig_main = plt.figure(figsize=(10, 6))
        draw_main(fig_main, tf, frames_to_plot, fn)
        fig_main.canvas.manager.set_window_title("PKL viewer (plots)")

        # ---- control figure (controls only) ----
        fig_ctrl = plt.figure(figsize=(6, 2.2))
        fig_ctrl.canvas.manager.set_window_title("Controls (doesn't block plots)")
        fig_ctrl.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        ax = fig_ctrl.add_axes([0, 0, 1, 1]); ax.axis("off")

        ax_box = fig_ctrl.add_axes([0.02, 0.60, 0.96, 0.25])
        tb = TextBox(ax_box, "frames:", initial=",".join(map(str, frames_to_plot)))

        ax_upd = fig_ctrl.add_axes([0.02, 0.25, 0.18, 0.25])
        ax_keep = fig_ctrl.add_axes([0.22, 0.25, 0.18, 0.25])
        ax_rej  = fig_ctrl.add_axes([0.42, 0.25, 0.18, 0.25])
        ax_skip = fig_ctrl.add_axes([0.62, 0.25, 0.18, 0.25])
        ax_quit = fig_ctrl.add_axes([0.82, 0.25, 0.16, 0.25])

        b_upd  = Button(ax_upd, "Update")
        b_keep = Button(ax_keep, "KEEP (Y)")
        b_rej  = Button(ax_rej, "REJECT (N)")
        b_skip = Button(ax_skip, "SKIP (S)")
        b_quit = Button(ax_quit, "Quit")

        def do_update(event=None):
            frs = parse_frames(tb.text, n_frames)
            if frs is None:
                print("Bad frames list. Example: 0,50,100,200")
                return
            draw_main(fig_main, tf, frs, fn)

        def close_all():
            plt.close(fig_ctrl)
            plt.close(fig_main)

        def do_keep(event=None):
            decision["val"] = "keep"
            close_all()

        def do_reject(event=None):
            decision["val"] = "reject"
            close_all()

        def do_skip(event=None):
            decision["val"] = "skip"
            close_all()

        def do_quit(event=None):
            decision["val"] = "quit"
            close_all()

        def on_key(event):
            k = event.key.lower()
            if k == "y": do_keep()
            elif k == "n": do_reject()
            elif k == "s": do_skip()

        b_upd.on_clicked(do_update)
        b_keep.on_clicked(do_keep)
        b_rej.on_clicked(do_reject)
        b_skip.on_clicked(do_skip)
        b_quit.on_clicked(do_quit)
        fig_ctrl.canvas.mpl_connect("key_press_event", on_key)
        fig_main.canvas.mpl_connect("key_press_event", on_key)

        # show both; control window不挡图
        plt.show()

        if decision["val"] == "keep":
            shutil.copy2(path, os.path.join(KEEP_FOLDER, fn))
            print("[KEEP]", fn)
        elif decision["val"] == "reject":
            print("[REJECT]", fn)
        elif decision["val"] == "skip":
            print("[SKIP]", fn)
        elif decision["val"] == "quit":
            print("[QUIT]")
            break

if __name__ == "__main__":
    review_folder()
