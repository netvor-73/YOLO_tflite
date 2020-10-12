from yolo_keras import YOLOv3, load_darknet_weights
import cv2
import numpy as np
import tensorflow as tf

yolo = YOLOv3(classes=80)

load_darknet_weights(yolo, 'yolov3.weights')

class_names = [c.strip() for c in open('coco.names', 'r').readlines()]

img_raw = cv2.imread('person.jpg')

image_shape = img_raw.shape[:2]

def get_input_to_network(img, input_dim=320):
    img = cv2.resize(img, (input_dim, input_dim), interpolation = cv2.INTER_CUBIC)
    img_ =  img[:,:,::-1].copy()  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = tf.convert_to_tensor(img_, dtype=tf.float32)
    return img_

blob = get_input_to_network(img_raw)

# print(blob.shape)

boxes, scores, classes, nums = yolo(blob)

print(nums)

print(f'classes: {classes.shape}: ', f'boxes: {boxes.shape}')
print(scores.shape)

boxes, scores, classes = boxes[0], scores[0], classes[0]

for index in range(nums[0].numpy().astype('uint32')):

    top = boxes[index, 0] * image_shape[1]
    left = boxes[index, 1] * image_shape[0]
    bottom = boxes[index, 2] * image_shape[1]
    right =  boxes[index, 3] * image_shape[0]

    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(img_raw.shape[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(img_raw.shape[0], np.floor(right + 0.5).astype('int32'))

    # print(class_names[classes[index]], (left, top), (right, bottom), scores[index])

    cv2.rectangle(img_raw, (top, left), (bottom, right), 128, 2)
    # cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)

cv2.imwrite('something.jpg', img_raw)
cv2.imshow("Image", img_raw)
cv2.waitKey(0)
cv2.destroyAllWindows()
