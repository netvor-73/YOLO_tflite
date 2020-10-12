from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import cv2
import numpy as np
import tensorflow as tf
import copy

vs = cv2.VideoCapture(0)

(W, H) = (None, None)

def get_input_to_network(img, input_dim=320):
    img = cv2.resize(img, (input_dim, input_dim), interpolation = cv2.INTER_CUBIC)
    img_ =  img[:,:,::-1].copy()  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]#/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = tf.convert_to_tensor(img_, dtype=tf.float32)
    return img_

################ model initialization ############################

interpreter = tf.lite.Interpreter(model_path='final.tflite')

input_index = interpreter.get_input_details()[0]['index']
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

LABELS = open('coco.names').read().strip().split("\n")

while True:
    grabbed, frame = vs.read()

    # print(cv2.imread('person.jpg'))

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # img = np.ascontiguousarray(copy.deepcopy(frame), dtype=np.float32)

    # img = cv2.imread('person.jpg')
    # cv2.imwrite('frame.jpg', frame)

    blob = get_input_to_network(frame)

    interpreter.set_tensor(input_index, blob)
    interpreter.invoke() # causing segmentation fault !

    boxes = interpreter.get_tensor(output_details[0]['index'])
    scores = interpreter.get_tensor(output_details[1]['index'])
    classes = interpreter.get_tensor(output_details[2]['index'])
    nums = interpreter.get_tensor(output_details[3]['index'])

    for c in range(int(nums[0])):
        print(LABELS[int(classes[0][c])])

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()