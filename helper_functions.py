import cv2
import numpy as np
from math import sqrt

img = []
ix, iy = 0, 0

def draw_circle(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        ix, iy = x, y


def get_points(image, numOfPoints, image_size=(800, 800)):
    global img
    img = image.copy()
    img = cv2.resize(img, image_size)
    width, height = image.shape[:2]
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_circle)
    points = []
    print("Press a for add point : ")
    while len(points) != numOfPoints:
        cv2.imshow("image", img)
        k = cv2.waitKey(1)
        if k == ord('a'):
            points.append([int(ix), int(iy)])
            cv2.circle(img, (ix, iy), 3, (0, 0, 255), -1)
    cv2.destroyAllWindows()
    return np.float32(points)


def create_model(config, weights):
    model = cv2.dnn.readNetFromDarknet(config, weights)
    backend = cv2.dnn.DNN_BACKEND_CUDA
    target = cv2.dnn.DNN_TARGET_CUDA
    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    return model


def get_output_layers(model):
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    return output_layers


def blob_from_image(image, target_size):
    blob = cv2.dnn.blobFromImage(image, 1 / 255., target_size, [0, 0, 0], 1, crop=False)
    return blob


def predict(blob, model, output_layers):
    model.setInput(blob)
    outputs = model.forward(output_layers)
    return outputs


def get_image_boxes(outputs, image_width, image_height, classes, confidence_threshold=0.5, nms_threshold=0.4):
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            class_name = classes[class_id]
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_name == 'person':
                cx, cy, width, height = (
                            detection[0:4] * np.array([image_width, image_height, image_width, image_height])).astype(
                    "int")
                x = int(cx - width / 2)
                y = int(cy - height / 2)
                boxes.append([x, y, int(width), int(height), cx, cy])
                confidences.append(float(confidence))
    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    #print("NMS indices: ", nms_indices)
    return [boxes[ind] for ind in nms_indices.flatten()]


def compute_point_perspective_transformation(matrix, boxes):
    # downoid = (x1 + w//2, y1)
    list_downoids = [[box[0] + box[2] // 2, box[1]] for box in boxes]
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return np.array(transformed_points_list).astype('int')


def get_red_green_boxes(distance_allowed, birds_eye_points, boxes):
    red_boxes = list()
    green_boxes = list()
    red_bev_lines = list()
    # red_lines = []

    # newboxes is a list contains tuples, each tuple contains (x1, y1, w, h, track_id, bev_x, bev_y)
    new_boxes = [tuple(box) + tuple(result) for box, result in zip(boxes, birds_eye_points)]

    for i in range(0, len(new_boxes) - 1):
        for j in range(i + 1, len(new_boxes)):
            bev_xi, bev_yi = new_boxes[i][5:]
            bev_xj, bev_yj = new_boxes[j][5:]
            distance = eucledian_distance([bev_xi, bev_yi], [bev_xj, bev_yj])
            #print("New boxes p1: {} , p2: {} ".format(new_boxes[i][4:6], new_boxes[j][4:6]))
            if distance < distance_allowed:
                #cv2.line(frame, (cxi, cyi), (cxj, cyj), (0,0,255), 2)
                track_id = new_boxes[j][4]
                bev1, bev2 = [bev_xi, bev_yi], [bev_xj, bev_yj]
                # bb1, bb2 = new_boxes[i][4:6], new_boxes[j][4:6]
                red_boxes.append(new_boxes[i])
                red_boxes.append(new_boxes[j])
                red_bev_lines.append(tuple([bev1, bev2]))
                # red_lines.append([bb1, bb2, track_id])

    green_boxes = list(set(new_boxes) - set(red_boxes))
    red_boxes = list(set(red_boxes))
    return (green_boxes, red_boxes, red_bev_lines)


def eucledian_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_birds_eye_view_image(green_box, red_box, eye_view_height, eye_view_width, red_bev_lines, image):
    blank_image = image
    blank_image = cv2.imread('background.png')
    
    num_risk = 0
    num_safe = 0
    if (green_box is not None) and (red_box is not None) and (red_bev_lines is not None):
        num_risk = len(red_box)
        num_safe = len(green_box)
        for green_point in green_box:
            cv2.circle(blank_image, tuple([green_point[5], green_point[6]]), 20, (0, 255, 0), -5)
            cv2.putText(blank_image,str(green_point[4]), tuple([green_point[5] + 10, green_point[6] - 10]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        for red_point in red_box:
            cv2.circle(blank_image, tuple([red_point[5], red_point[6]]), 20, (0, 0, 255), -5)
            cv2.putText(blank_image, str(red_point[4]), tuple([red_point[5] + 10, red_point[6] - 10]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
        for red_pair in red_bev_lines:
            #print("P1: {}, P2: {}".format(tuple(pp[0]), tuple(pp[1]) ) )
            cv2.line(blank_image, red_pair[0], red_pair[1], (0, 0, 255), 10)
            # cv2.putText(blank_image,str(bev_point[2]), tuple(bev_point[0], bev_point[1] - 5), cv2.FONT_HERSHEY_PLAIN, 0.5,(255, 255, 255), 2)
    
    cv2.putText(blank_image, "RISK: "+ str(num_risk), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(blank_image, "SAFE: "+ str(num_safe), (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    
    blank_image = cv2.resize(blank_image, (eye_view_width, eye_view_height))
    return blank_image


def get_red_green_box_image(new_box_image, green_box, red_box):
    if (green_box is not None) and (red_box is not None):
        for green_bbox in green_box:
            cv2.rectangle(new_box_image, (green_bbox[0], green_bbox[1]), (green_bbox[0] + green_bbox[2], green_bbox[1] + green_bbox[3]), (0, 255, 0), 2)
        for red_bbox in red_box:
            cv2.rectangle(new_box_image, (red_bbox[0], red_bbox[1]), (red_bbox[0] + red_bbox[2], red_bbox[1] + red_bbox[3]), (0, 0, 255), 2)
            # cv2.line(new_box_image, ppp[0], ppp[1], (0, 0, 255), 2)
        #for ppp in pointpp:
            #print("p1: {}, p2: {}".format(ppp[0], ppp[1]))
            #cv2.line(new_box_image, ppp[0], ppp[1], (0, 0, 255), 2)
    return new_box_image