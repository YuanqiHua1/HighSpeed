"""
This script calculate different variables from the pkl file which contains the tail coordinates tracked by joetailfit script.

Some parameters should be filled up as required (Line 13-18). Max. deflection angle to be consider a swmming movement in Line 204 &208.

The output will be located in the same folder than the input file (pklfile, Line 18). The first row will be the headers
for each variable, the second row will be the value of each variable written in different columns.
"""
from babel.plural import to_python

"""PARAMETERS TO FILL UP"""

Larva_ID = "1"
Genotype = "wt1 MTZ control"
fps = 750
decimals_required = 3
frame_for_smoothing = 3
pklfile = r'C:\Users\huayuanqi\OneDrive - Uppsala universitet\Desktop\PhD\PhD\high speed\scripy\ts3_000052-1.pkl'

import pickle
import numpy as np
import os.path
import matplotlib.pyplot as plt


""" TO OPEN tHE pkl FILE:"""
with open(pklfile, 'rb') as d:
    tf = pickle.load(d)

#print len(tf) # to print the number of frames in tf (tailfit array)
#print "shape tf array", np.shape(tf) # to print the dimensions of tf (tailfit array)
#print tf[:4]


"""TO SMOOTH THE TAIL_FIT DATA in both axis (x,y):"""
from astropy.convolution import convolve, Box1DKernel
tf_swapped = np.swapaxes(tf,0,1)
#print "shape tf_swapped array", np.shape(tf_swapped)
#print tf_swapped[:,10]
tf_swapped_smoothed = [] #creating a empty list where to store the smoothed data we'll generate in the following loop
for tailpoint in tf_swapped:
    x = tailpoint[:, 0] #selecting only the x axis
    x_smoothed = convolve(tailpoint[:,0], Box1DKernel(3)) #apply moving average every 15 frame by Boxcar filter
    x_smoothed_cropped = x_smoothed[10:-10]  # discard the first and last 10 frames from the smoothed data
    y = tailpoint[:, 1] #selecting only the y axis
    y_smoothed = convolve(tailpoint[:,1], Box1DKernel(3))  # apply moving average every 15 frame by Boxcar filter
    y_smoothed_cropped = y_smoothed[10:-10]  # discard the first and last 10 frames from the smoothed data
   # 251217 revised by YH, zip is for python2, np.array is replaced to adjust to python3:
    # smoothed_coordinates = zip(x_smoothed_cropped, y_smoothed_cropped) #zipping the smoothed_cropped data from both axis
    smoothed_coordinates = np.array(list(zip(x_smoothed_cropped, y_smoothed_cropped)))

    tf_swapped_smoothed.append(smoothed_coordinates) #including the smoothed coordinates in the empty array

    """
    plt.plot(x, 'y') #plotting original x data in yellow
    plt.plot(x_smoothed_cropped, 'b')  # plotting smoothed x data in blue
    plt.show()
    plt.plot(y, 'r') #plotting original y data in red
    plt.plot(y_smoothed_cropped, 'g')  #plotting smoothed y data in green
    plt.show()
    """

#print "smoothed coordinates", np.shape(smoothed_coordinates)
#print "tf_swapped_smoothed shape", np.shape(tf_swapped_smoothed)
smoothed_tf = np.swapaxes(tf_swapped_smoothed,0,1)
#print "smoothed_tf", np.shape(smoothed_tf)


"""
plt.plot(smoothed_tf[:, -1,0], 'b')  # smoothed tailtip data in blue
plt.plot(smoothed_tf[:, 0,0], 'r')  # smoothed tail-basedata in red
plt.plot(smoothed_tf[:, 9,0], 'g')  # smoothed mid-tail in green
plt.show()
"""


"""TO CALCULATE THE DEFLECTION ANGLE of the tail tip per each frame:"""
dx = [frame[-1,0] - frame[0,0] for frame in smoothed_tf] #distance in the x axis from tail baseline to the tail tip for each frame
dy = [frame[-1,1] - frame[0,1] for frame in smoothed_tf] #distance in the y axis from tail baseline to the tail tip for each frame
deflection = np.degrees(np.arctan(np.divide(dx, dy))) #tangent (in degrees) of the tail tip ditance in axis x and axis y from the basal tip point
#print "max. deflection to the right (in degrees):", max(deflection), "& frame of max. deflection to the right:", np.argmax(deflection)+1
#print "max. deflection to the left (in degrees):", min(deflection), "& frame of max. deflection to the left:", np.argmin(deflection)+1

max_deflection = round(max(abs(deflection)),decimals_required)
frame_max_deflection = np.argmax(abs(deflection))+1


"""TO CALCULATE THE CURVATURE ALONG THE TAIL per each frame:"""
def tail_segment(P1, P2):
    return np.degrees(np.arctan2(P2[1] - P1[1], P2[0] - P1[0])) #P(Y,X) to get the smallest angle to the reference axis

atan_segments = []
# for frame in np.array(zip(smoothed_tf[:, :-1], smoothed_tf[:, 1:])): # loop to calculate the angles of each segment per each frame
for P1s, P2s in zip(smoothed_tf[:, :-1], smoothed_tf[:, 1:]):
    atan_segments_frame = []
    for P1, P2 in zip(P1s, P2s):
   # for i in frame[0]:
    #    for j in frame[1]:
     #       P1 = i
    #        P2 = j
        atan_segments_frame.append(tail_segment(P1, P2))
    atan_segments.append(atan_segments_frame)

angles_segments = np.array([(angle_frame[0] - angle_frame[1]) for angle_frame in zip(np.array(atan_segments)[:, :-1], np.array(atan_segments)[:, 1:])]) # getting the angles between consecutive segments per frame
angles_between_segments = [[abs(angle) if abs(angle)<= 180 else abs(360-abs(angle)) for angle in frame] for frame in angles_segments] # converting all the angles in absolute values
k = [sum(frame[:]) for frame in np.array(angles_between_segments)] #Curvature (k), addition of all the angles between segments per frame

max_curvature = round(max(k), decimals_required)
frame_max_curvature = np.argmax(k)+1
segment_max_angle_when_max_curvature = np.argmax(angles_between_segments[np.argmax(k)])
#print np.argmax(angles_between_segments, 1)


"""
#TO PICK THE BEGINING AND END OF THE MOVEMENT & DURATION
Increment_k = [abs(frame[0] - frame[1]) for frame in zip(k[:-1], k[1:])] #Calculate the difference in curvature from frame to frame
#print Increment_k

before_movement_starts = [] #generate an empty array where to store the frames before the movement happens
for item in Increment_k:
    if item > 2: #I have established changes above 0.5 degrees as the threshold for movements
        break
    #print item, k.index(item)+1
    before_movement_starts.append(Increment_k.index(item)+1)
print "movement starts at frame:", before_movement_starts[-1]

movement_ends = []
for item in Increment_k[::-1]: #for item in the frame-reversed curvature array
    if item > 2: #I have established cahnges above 0.5 degrees as the threshold for movements
        break
    #print item, k.index(item)+1
    movement_ends.append(Increment_k.index(item)+1)
print "movement ends at frame:", movement_ends[-1]

Movement_duration = movement_ends[-1] - before_movement_starts[-1]
print "movement duration (in frames):", Movement_duration
"""


"""TO FIND THE HALF BEATS by https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html#scipy.signal.find_peaks"""
# Peaks: toward the right:
smoothed_tf_x = [frame[-1,0] for frame in smoothed_tf] #selecting only the x coordenate for the tailtip
smoothed_tf_x = np.asarray(smoothed_tf_x) #To make smoothed_tf_x integer scalars
from scipy.signal import find_peaks
peaks_right, properties_peaks_right = find_peaks(smoothed_tf_x, distance=10, prominence=20, width=1, rel_height=0.5)
# Trough: toward the left:
inverted_smoothed_tf_x = -smoothed_tf_x
peaks_left, properties_peaks_left = find_peaks(inverted_smoothed_tf_x, distance=10, prominence=20, width=1, rel_height=0.5)

Dict_right = dict({'Peaking_frame':peaks_right}, **properties_peaks_right)
Dict_left = dict({'Peaking_frame':peaks_left}, **properties_peaks_left)

#print "half beats to the right at frames:",(peaks_right)
#print "half beats to the left at frames:",(peaks_left)

mean_curvature = round(np.mean(k[int(min(properties_peaks_right["left_ips"][0], properties_peaks_left["left_ips"][0])):int(max(properties_peaks_right["left_ips"][-1], properties_peaks_left["left_ips"][-1]))]), decimals_required)
#print "Tail path per beat:", properties_peaks_right["prominences"], properties_peaks_left["prominences"]
Longest_tail_trajectory = round(np.max(np.concatenate((properties_peaks_right["prominences"],properties_peaks_left["prominences"]))),decimals_required)
Mean_tail_trajectory = round(np.mean(np.concatenate((properties_peaks_right["prominences"],properties_peaks_left["prominences"]))),decimals_required)
Cumulative_tail_trajectory = round(np.sum(np.concatenate((properties_peaks_right["prominences"],properties_peaks_left["prominences"]))),decimals_required)


"""DURATION of THE MOVEMENT"""
Movement_duration = round(((max(properties_peaks_right["right_ips"][-1], properties_peaks_left["right_ips"][-1])) - (min(properties_peaks_right["left_ips"][0], properties_peaks_left["left_ips"][0])))/fps,decimals_required)
#print "first frame of the movement:", min(properties_peaks_right["left_ips"][0], properties_peaks_left["left_ips"][0])
#print "last frame of the movement:", max(properties_peaks_right["left_ips"][-1], properties_peaks_left["left_ips"][-1])


"""TAIL HALF BEAT FREQUENCY"""
Nr_half_beats = peaks_right.size + peaks_left.size # Total number of half beats (to the right and to the left)
Freq_half_beats = round(float(Nr_half_beats)/(Movement_duration),decimals_required)


"""DEFLECTION IN THE PEAKS"""
Deflection_in_peaks =[]
Deflection_in_peaks_right = []
Deflection_in_peaks_left = []
Curvature_in_peaks =[]
for element in peaks_right:
    Deflection_in_peaks.append(deflection[element])
    Deflection_in_peaks_right.append(deflection[element])
    Curvature_in_peaks.append(k[element])
for element in peaks_left:
    Deflection_in_peaks.append(deflection[element])
    Deflection_in_peaks_left.append(deflection[element])
    Curvature_in_peaks.append(k[element])

Mean_deflection_peaks = round(np.mean([abs(deflection_element) for deflection_element in Deflection_in_peaks]), decimals_required)
Mean_curvature_peaks = round(np.mean(Curvature_in_peaks), decimals_required)


"""TURN BIAS (RIGHT/LEFT)"""
bias_right = 0
bias_left = 0
for element in Deflection_in_peaks:
    if element > 0:
        bias_right = bias_right +1
    if element < 0:
        bias_left = bias_left + 1
Bias_RL = round(float(bias_right)/float(bias_left), decimals_required)


"""SELECTING ONLY SWIMMING -  Swimming duration"""
Left_ips_Swimming_right = []
Right_ips_Swimming_right = []
Left_ips_Swimming_left = []
Right_ips_Swimming_left = []
for element in Deflection_in_peaks_right:
    if abs(element) < 50: # considering deflections 50 degrees only happen during the C-turn in the escape
        Left_ips_Swimming_right.append(properties_peaks_right["left_ips"][Deflection_in_peaks_right.index(element)])
        Right_ips_Swimming_right.append(properties_peaks_right["right_ips"][Deflection_in_peaks_right.index(element)])
for element in Deflection_in_peaks_left:
    if abs(element) < 50: # considering deflections above 50 degrees only happen during the C-turn in the escape
        Left_ips_Swimming_left.append(properties_peaks_left["left_ips"][Deflection_in_peaks_left.index(element)])
        Right_ips_Swimming_left.append(properties_peaks_left["right_ips"][Deflection_in_peaks_left.index(element)])
Swimming_duration = round((max((Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1]))- min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0]))/fps, decimals_required)


"""VARIABLES DURING SWIMMING"""
#HALF BEATS & TAIL HALF BEAT FREQUENCY:
peaks_swimming_right = []
peaks_swimming_left = []
for frame in peaks_right:
    if frame > min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0]):
        if frame < max((Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1])):
            peaks_swimming_right.append(frame)
for frame in peaks_left:
    if frame > min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0]):
        if frame < max((Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1])):
            peaks_swimming_left.append(frame)
Nr_half_beats_swimming = np.size(peaks_swimming_right)+np.size(peaks_swimming_left)
freq_half_beats_swimming = round(Nr_half_beats_swimming/Swimming_duration, decimals_required)

#CURVATURE
k_swimming = k[int(min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0])):int(max((Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1])))]
mean_curvature_swimming = round(np.mean(k_swimming), decimals_required)
max_curvature_swimming = round(max(k_swimming), decimals_required)
frame_max_curvature_swimming = np.argmax(k_swimming)+1
segment_max_angle_when_max_curvature_swimming = np.argmax(angles_between_segments[np.argmax(k_swimming)])

#DEFLECTION
deflection_swimming = deflection[int(min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0])):int(max((Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1])))]
max_deflection_swimming = round(max(deflection_swimming), decimals_required)
frame_max_deflection_swimming = np.argmax(deflection_swimming)+1
mean_deflection_swimming = round(np.mean(deflection_swimming), decimals_required)

Deflection_in_peaks_swimming =[]
#Deflection_in_peaks_right = []
#Deflection_in_peaks_left = []
Curvature_in_peaks_swimming =[]
for element in peaks_swimming_right:
    Deflection_in_peaks_swimming.append(deflection[element])
    #Deflection_in_peaks_right.append(deflection[element])
    Curvature_in_peaks_swimming.append(k[element])
for element in peaks_left:
    Deflection_in_peaks_swimming.append(deflection[element])
    #Deflection_in_peaks_left.append(deflection[element])
    Curvature_in_peaks_swimming.append(k[element])
mean_deflection_swimming_peaks = round(np.mean([abs(deflection_element) for deflection_element in Deflection_in_peaks_swimming]), decimals_required)
mean_curvature_swimming_peaks = round(np.mean([abs(curvature_element) for curvature_element in Curvature_in_peaks_swimming]), decimals_required)

#BIAS TO TURN RIGHT OR LEFT
bias_swimming_right = 0
bias_swimming_left = 0
for element in Deflection_in_peaks_swimming:
    if element > 0:
        bias_swimming_right = bias_right +1
    if element < 0:
        bias_swimming_left = bias_left + 1
Bias_RL_swimming = round(float(bias_swimming_right)/float(bias_swimming_left), decimals_required)


""" To save the arrays in txt format. Modify the filepath of the file with the extension .txt"""
def numpy_to_python(x):
    return float(x) if isinstance(x, np.generic) else x

variables = (
                tuple(map(numpy_to_python,(
                    'Larva ID', 'Genotype', 'Movement duration', 'Nr. half beats', 'Freq. half beats','Max. deflection', 'Frame at max deflection', 'Mean deflection during halft beats',
              'Longest tail trajectory among half beats', 'Cumulative tail trajectory', 'Max. curvature', 'Frame at max curvature', 'Tail segment with the max. angle at max. curvature',
              'Mean curvature', 'Mean curvature during beats', 'Turning bias to R/L', 'Swimming duration', 'Nr of half beats during swimming', 'Freq. half beating during swimming',
              'Max. deflection during swimming', 'Frame at max deflection swimming', 'Mean deflection during swimming beats', 'Max. curvature swimming', 'Frame at max curvature when swimming',
              'Tail segment with max angle at max curvature', 'Mean curvature during swimming', 'Mean curvature during swimming beats', 'Turning bias to R/L during swimming'
                ))),
             tuple(map(numpy_to_python,(
                 Larva_ID, Genotype, Movement_duration, Nr_half_beats, Freq_half_beats, max_deflection, frame_max_deflection, Mean_deflection_peaks, Longest_tail_trajectory, Cumulative_tail_trajectory,
              max_curvature, frame_max_curvature, segment_max_angle_when_max_curvature, mean_curvature, Mean_curvature_peaks, Bias_RL, Swimming_duration, Nr_half_beats_swimming, freq_half_beats_swimming,
              max_deflection_swimming, frame_max_deflection_swimming, mean_deflection_swimming_peaks, max_curvature_swimming,  frame_max_curvature_swimming, segment_max_angle_when_max_curvature_swimming, mean_curvature_swimming,
              mean_curvature_swimming_peaks, Bias_RL_swimming
             )))
)

savefilepath = os.path.splitext(pklfile)[0] + '.txt'
output=open(savefilepath,"w")
output.write(str('\n'.join(map(str, variables))).replace('(', '').replace(')', ''))
output.flush()
output.close()


"""
#PLOTS:
plt.suptitle('Curvature, Deflection & Half Beats')
plt.subplot(3,1,1)
plt.ylabel("Curvature (deg)")
plt.plot(k, 'b')
plt.subplot(3,1,2)
plt.ylabel("Deflection (deg)")
plt.plot(deflection, 'g')
plt.xlabel("Time (in Frames)")
plt.subplot(3,1,3)

plt.plot(smoothed_tf_x)
plt.plot(peaks_right, smoothed_tf_x[peaks_right], "x")
plt.vlines(x=peaks_right, ymin=(smoothed_tf_x[peaks_right] - properties_peaks_right["prominences"]), ymax = smoothed_tf_x[peaks_right], color = "C3")
plt.hlines(y=properties_peaks_right["width_heights"], xmin=properties_peaks_right["left_ips"], xmax=properties_peaks_right["right_ips"], color = "C1")
plt.plot(peaks_left, smoothed_tf_x[peaks_left], "x")
plt.vlines(x=peaks_left, ymin=smoothed_tf_x[peaks_left], ymax = smoothed_tf_x[peaks_left] + properties_peaks_left["prominences"], color = "C3")
plt.hlines(y=(-properties_peaks_left["width_heights"]), xmin=properties_peaks_left["left_ips"], xmax=properties_peaks_left["right_ips"], color = "C2")
plt.hlines(xmin=properties_peaks_left["left_ips"][0],xmax=properties_peaks_left["right_ips"][-1], y=max(smoothed_tf_x[peaks_right]) )
plt.hlines(xmin=min(Left_ips_Swimming_right[0], Left_ips_Swimming_left[0]),xmax=max(Right_ips_Swimming_right[-1], Right_ips_Swimming_left[-1]), y=max(smoothed_tf_x[peaks_right])-5, color = "C8" )
plt.ylabel("Tailtip in x axis")
plt.xlabel("Time (in Frames)")
plt.show()



print "#", Larva_ID
print Genotype
print "Movement duration (in sec):", Movement_duration
print "Nr of half beats:",Nr_half_beats
print "Freq. half beating (in Hz):", Freq_half_beats
print "Max. deflection (in degrees):", max_deflection
print "Mean deflection during halft beats (in degrees):", Mean_deflection_peaks
print "Longest tail trajectory among half beats (in pixels):", Longest_tail_trajectory
#print "Mean tail trajectory (in pixels):", Mean_tail_trajectory
print "Cumulative tail trajectory (in pixels):", Cumulative_tail_trajectory
print "Max. curvature", max_curvature
print "Tail segment with the max. angle when the max. curvature happen", segment_max_angle_when_max_curvature
print "Mean curvature", mean_curvature
print "Mean curvature during beats:", Mean_curvature_peaks
print "Turning bias to R/L:", Bias_RL
print ''
print "Swimming duration (in sec):", Swimming_duration
print "Nr of half beats:", Nr_half_beats_swimming
print "Freq. half beating during swimming (in Hz):", freq_half_beats_swimming
print "Max. deflection (in degrees) during swimming:", max_deflection_swimming
print "Mean deflection during swimming beats (in degrees):", mean_deflection_swimming_peaks
print "Max. curvature during swimming:", max_curvature_swimming
print "Mean curvature during swimming:", mean_curvature_swimming
print "Mean curvature during swimming beats:", mean_curvature_swimming_peaks
print "Turning bias to R/L during swimming:", Bias_RL_swimming
"""
