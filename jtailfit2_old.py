from __future__ import division, print_function, absolute_import
import cv2
import numpy as np
import time
import scipy
import scipy.ndimage
import scipy.spatial.distance


def getbackgroundsign(frame, point, pointhalfwindow=3):
    """
    Evalutes whether fish or background is brighter
    Returns 1 if the fish is brighter, -1 otherwise
    Uses a historgram to find the frame brightness, and
    """
    # find background - 10 bin hist of frame, use most common as background
    # find fish luminosity - area around point
    assert frame.ndim == 2

    pw = pointhalfwindow
    fish = frame[point[1] - pw:point[1] + pw, point[0] - pw:point[0] + pw].mean()
    hist = np.histogram(frame, 10, (0, 255))  # TODO check depth?
    background = hist[1][hist[0].argmax()] / 2 + hist[1][min(hist[0].argmax() + 1, len(hist[0]))] / 2

    return np.sign(fish - background)



#         # @staticmethod
#         # def frompath(path):
#         #     maketailfit object
#         #     init
#         #     runtf


def tailfit_simple(vid, start_point, end_point, num_points=20, display=True, variabledelay=True,arcradius = 60):
    blursize = 7 #size in pixels for a 2D blur
    # arcradius = 60 修改
    display_point_color = (0, 0, 255)

    taillength = scipy.spatial.distance.euclidean(start_point, end_point)
    tailpoint_spacing = taillength / num_points

    start_vector = end_point - start_point
    start_vector = start_vector / np.linalg.norm(start_vector)

    frame_fit = np.zeros((num_points, 2))
    # first_frame = True
    fitted_tail = []  # TODO preallocate

    print("Starting tailfit on:  ", vid.filepath)
    print("fps is: ", vid.FPS, " and frame count is: ", len(vid))



    if display:
        cv2.namedWindow("frame_display")
        cv2.moveWindow("frame_display", 0, 0)

    starttime = time.time()

    for framenum, frame in enumerate(vid):
        if display:
            frame_display = frame.copy()
        if frame.ndim == 3:  # if video frame is color
            frame = frame[..., 1]  # take 2nd, green channel
        frame = cv2.boxFilter(frame, -1, (blursize, blursize))  # TODO is this the right size, have tunable?
        if framenum == 0:
            backgroundsign = getbackgroundsign(frame, start_point)

        guess_vector = start_vector
        current = start_point

        for count in range(num_points):
            if count > 0:
                guess_vector = guess_vector / np.linalg.norm(guess_vector)

            # from guess_vector, find the middle angle in radians
            arccenter = np.arctan2(*guess_vector)
            lin = np.linspace(-np.pi * .3 + arccenter, np.pi * .3 + arccenter, 40)

            xs = current[0] + arcradius * np.sin(lin) - guess_vector[0] * (arcradius - tailpoint_spacing)
            ys = current[1] + arcradius * np.cos(lin) - guess_vector[1] * (arcradius - tailpoint_spacing)
            x_indices, y_indices = xs.astype(int), ys.astype(int)

            if max(y_indices) >= frame.shape[0] or min(y_indices) < 0 or max(x_indices) >= frame.shape[1] or min(
                    x_indices) < 0:
                y_indices = np.clip(y_indices, 0, frame.shape[0] - 1)
                x_indices = np.clip(x_indices, 0, frame.shape[1] - 1)
                print("Tail got too close to the edge of the frame, clipping search area!")

            guess_slice = frame[y_indices, x_indices]  # the frame is transposed compared to what might be expected

            guess_slice = backgroundsign * guess_slice

            # plt.plot(guess_slice, 'r')
            # plt.show()

            # adpative background brightness/baseline
            # hist = np.histogram(guess_slice, 10)
            # guess_slice = guess_slice - guess_slice[
            #     ((hist[1][hist[0].argmax()] <= guess_slice) & (guess_slice < hist[1][hist[0].argmax() + 1]))].mean()

            # sguess = scipy.ndimage.filters.percentile_filter(guess_slice, 50, 20)
            # sguess = scipy.ndimage.filters.uniform_filter1d(guess_slice, blurfactor)

            # plt.plot(sguess);
            # plt.show()

            # result_index = sguess.argmax()  # - results.size / 2 + guess_slice.size / 2

            normcumslice = np.cumsum(guess_slice - guess_slice[:3].mean() * .5 - guess_slice[-3:].mean() * .5)
            result_index = (np.abs(normcumslice - normcumslice[-1] * .5)).argmin()

            # plt.plot(np.cumsum(guess_slice-guess_slice[0]));
            # plt.axvline(result_index)
            # plt.show()

            newpoint = np.array([x_indices[result_index], y_indices[result_index]])

            if display:
                cv2.circle(frame_display, (int(newpoint[0]), int(newpoint[1])), 2, display_point_color)
                frame_display[y_indices, x_indices] = 0

            frame_fit[count, :] = newpoint

            if count > 0:
                guess_vector = newpoint - current
            current = newpoint

        fitted_tail.append(np.copy(frame_fit))
        if display:
            cv2.putText(frame_display, str(framenum), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (225, 10, 20));
            # cv2.imshow("frame_display", frame) 修改
            cv2.imshow("frame_display", frame_display)
            if framenum == 0:
                delaytime = 1
            else:
                if variabledelay:
                    minlen = min([fitted_tail[-2].shape[0], fitted_tail[-1].shape[0]]) - 1
                    delaytime = int(min(max((np.abs(
                        (fitted_tail[-2][minlen, :] - fitted_tail[-1][minlen, :]) ** 2).sum() ** .5) ** 1.2 * 3 - 1, 1),
                                        500))
                else:
                    delaytime = 17
            cv2.waitKey(delaytime)

    print("Tailfit done in %.2f seconds" % (time.time() - starttime))
    return fitted_tail

#TODO
# convience methods, that runs init and runtf on a certain file/dir - reuse some of batch but switch to QT
# better handle if the tail runs out of the frame?
# class for algorithim, which can enumerate it's parameters and types (eventually)
# have option for multiprocessing? how to distribute, since adaptive basepoint require continuity
# support being run from a stream?
# robust lin reg and subtract to handle if the image has strong gradients
# ?use attrs lib to make a result object? or dataframe?
