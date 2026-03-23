import tailclick # manually pick points on an image (e.g., start and end of the tail).
from videowrapper import VideoWrapper # loads a video and provides easy access to its frames.
import jtailfit2_old # contains functions to analyze the tail movement from the video.
import pickle # allows saving Python objects to a file and loading them later.
import os.path
#Added by Ana dP: line 5 to use it in line 28
import numpy as np

if __name__ == "__main__":
    filepath = r'U:\YuanqiHua\High speed\260123 wt1 MTZ\AVI_out\wt1-MTZ-control-1_7.avi'
    # filepath = r"C:\holo_paper\figure 3\2p triggered position behavior\full\1171ChR2_t54.avi"

    vid = VideoWrapper(filepath)
    firstframe = vid.firstframe

    startpoint, endpoint = tailclick.picktwopoints(firstframe)
    # startpoint, endpoint = np.asarray([618, 314]), np.asarray([598, 712])

    # print startpoint, endpoint
    # print jtailfit2.getbackgroundsign(firstframe, startpoint)

    tf = jtailfit2.tailfit_simple(vid, startpoint, endpoint, display=True,arcradius = 60)

    # print (len(tf))
    # with open('test_tailfit.pkl', 'wb') as f:
    #     pickle.dump(tf, f)

    savefilepath = os.path.splitext(filepath)[0] + '.pkl'
    with open(savefilepath, 'wb') as f:
        pickle.dump(tf, f)

    #Added by Ana dP from line 27 to 29 in order to save the output