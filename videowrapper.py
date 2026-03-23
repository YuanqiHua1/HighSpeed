from __future__ import division, print_function, absolute_import
import cv2
import os


class CapPropVer2(object):
    """
    Provides interface compatibilty for opencv version 2
    """
    def __getattr__(self, item):
        if item.startswith('CAP_PROP'):
            print("cap")
        return cv2.cv.__getattribute__('CV_'+item)

if cv2.__version__.startswith('2'):
    capprop = CapPropVer2()
else:
    capprop = cv2

class VideoWrapper(object):
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            print("Error with videopath!")
            raise Exception("Path doesn't exist! %s " % self.filepath)
        self.cap = cv2.VideoCapture(self.filepath)
        if not self.cap.isOpened():
            print("Error with video!")
            print("    Filename exists, probably a codec issue")
            print("    FOURCC: ", self.cap.get(capprop.CAP_PROP_FOURCC))
            raise Exception('Issues opening video file! %s ' % self.filepath)

    def __next__(self):
        retcode, frame = self.cap.read()
        if retcode:
            return frame
        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.cap.get(capprop.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, key):
        if isinstance(key, slice):
            self.cap.set(capprop.CAP_PROP_POS_FRAMES, key.start)
            return [next(self) for i in range(key.stop - key.start)]
            # TODO handle step!
        elif isinstance(key, int):
            self.cap.set(capprop.CAP_PROP_POS_FRAMES, key)
            return next(self)
        else:
            raise TypeError("Invalid argument type.")

    @property
    def FPS(self):
        return self.cap.get(capprop.CAP_PROP_FPS)

    @property
    def firstframe(self):
        firstframe = self[0]
        self.cap.set(capprop.CAP_PROP_POS_FRAMES, 0) #reset position to start
        return firstframe


if __name__ == "__main__":
    # filepath = r'I:\20150224_BEHAVIORDATA\20160122\hucnls6s1171ChR2_M6.avi'
    filepath = r'C:\joetailfit2_Improved\Experiments_ADPC\Maintained_swimming_ZF.avi.avi'

    vid = VideoWrapper(filepath)
    print(vid[4])
    print(len(vid[5:10]))
