import cv2
import numpy as np
from helper_functions import *



original_image_BGR = cv2.imread('MOT20_02_raw_frame_100.jpg')

original_image_RGB = cv2.cvtColor(original_image_BGR, cv2.COLOR_BGR2RGB)
#main_header = cv2.imread('templates/main_header.jpg')

image_width = original_image_RGB.shape[1]
image_height = original_image_RGB.shape[0]

original_image_BGR_copy = original_image_BGR.copy()
original_image_RGB_copy = original_image_RGB.copy()

#print('image Shape', original_image_RGB.shape)

# points for MOT20_02_raw.mp4 or MOT20_02_raw_frames
source_points = np.float32([[142., 298.],
                           [784., 315.],
                           [811., 371.],
                           [ 82., 347.]])

# source_points = np.float32([[ 401.,  132.], [1492.,  106.], [1883.,  727.], [  35.,  763.]])

# for point in source_points:
#     point = list(map(int, point))
#     cv2.circle(original_image_RGB_copy, tuple(point), 8, (255, 0, 0), -1)

points = source_points.reshape((-1,1,2)).astype(np.int32)
cv2.polylines(original_image_RGB_copy, [points], True, (0,255,0), thickness=4)

src = source_points
dst = np.float32([(0.5, 0.78), # tl
                  (0.80, 0.78), # tr
                  (0.80, 0.84), # br
                  (0.5, 0.84)]) # bl
# dst = np.float32([(0.25,0.47), (0.78, 0.47), (0.78,0.77), (0.25,0.77)])

dst_size = (image_height, image_width)
dst = dst * np.float32(dst_size)
#dst=np.float32([(160.,885.6), (640, 885.6), (640,1000.6), (160,1000.6)])
print(dst)

H_matrix = cv2.getPerspectiveTransform(src, dst)
#print("The perspective transform matrix:")
#print(H_matrix)
print("Height", image_height)
print("Width", image_width)

warped = cv2.warpPerspective(original_image_RGB_copy, H_matrix, dst_size)

# for point in dst:
#     point = list(map(int, point))
#     cv2.circle(warped, tuple(point), 8, (255, 0, 0), -1)

warped = cv2.resize(warped, (960, int(960 * (1080/1920))))



original_image_RGB_copy = cv2.resize(original_image_RGB_copy, (600, 400))

cv2.imshow("Test", warped)
cv2.imshow("OG image", original_image_RGB_copy)
cv2.waitKey(0)