import cv2
import numpy as np
from helper_functions import *
import matplotlib.pyplot as plt
import time

# Source point calib trong calib.py
#########################################################################

#source_points = np.float32([[ 436.,  187.], [1016.,  264.], [ 901.,  487.], [ 248., 296.]]) #testclip 2
#source_points = np.float32([[142., 298.], [784., 315.], [811., 371.],[ 82., 347.]])  #MOT clip
source_points = np.float32([[ 401.,  132.], [1492.,  106.], [1883.,  727.], [  35.,  763.]]) #people.mp4

##########################################################################

points = source_points.reshape((-1,1,2)).astype(np.int32)
#cv2.polylines(original_image_RGB_copy, [points], True, (0,255,0), thickness=4)

src=source_points
# Scale point dst calib trong calib3.py
###########################################################################

#dst=np.float32([(0.2,0.82), (0.80, 0.82), (0.80,0.87), (0.2,0.87)]) # cho MOT clip
#dst=np.float32([(0.4,0.52), (0.80, 0.52), (0.80,0.67), (0.4,0.67)])  # cho testvideo2 clip
dst=np.float32([(0.25,0.47), (0.78, 0.47), (0.78,0.77), (0.25,0.77)]) #cho people.mp4

############################################################################

dst_size=(800,1080)
dst = dst * np.float32(dst_size)

H_matrix = cv2.getPerspectiveTransform(src, dst)

confidence_threshold = 0.5
nms_threshold = 0.5
min_distance = 35

      #LOAD VIDEO
#video = cv2.VideoCapture('videos/testvideo2.mp4')
#video = cv2.VideoCapture('videos/MOT20-02-raw.webm')
video = cv2.VideoCapture('videos/people.mp4')

class_names = []
with open("models/coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
fps = int(video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("res.mp4", fourcc, fps, (frame_width, frame_height))
while True:
    ret, frame = video.read()
    if not ret:
        break
    image_height, image_width = frame.shape[:2]
    cv2.polylines(frame, [points], True, (0, 255, 0), thickness=4)
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    warped = cv2.warpPerspective(frame, H_matrix, dst_size)
    t1 = time.time()
    classes, _ , boxes = model.detect(frame, confidence_threshold, nms_threshold)
    list_boxes = []

    for (classid, box) in zip(classes, boxes):
        if class_names[classid] != 'person': continue
        x, y, w, h = box
        center_x, center_y = int(x+w/2), int(y+h/2)
        list_boxes.append([x, y, w, h, center_x, center_y])
        
    birds_eye_points = compute_point_perspective_transformation(H_matrix, list_boxes)
    
    green_box, red_box, pp, ppp = get_red_green_boxes(min_distance, birds_eye_points, list_boxes)
    
    birds_eye_view_image = get_birds_eye_view_image(green_box, red_box,eye_view_height=image_height,eye_view_width=image_width//2, points=pp, image=warped)
    
    box_red_green_image = get_red_green_box_image(frame.copy(),green_box, red_box, ppp)
    
    combined_image = np.concatenate((birds_eye_view_image,box_red_green_image), axis=1)
    resize_combined_image = cv2.resize(combined_image, (1440, 540))
    
    fps = 1. / (time.time() - t1)
    cv2.putText(resize_combined_image,"FPS: " +str(round((fps), 2)),(530, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # out.write(resize_combined_image)
    cv2.imshow("Social Distance", resize_combined_image)
    if cv2.waitKey(1) == ord('q'): break

video.release()
out.release()
cv2.destroyAllWindows()