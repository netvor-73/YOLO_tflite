from numpy.core.fromnumeric import shape
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import tensorflow as tf
import pathlib
import numpy as np
import cv2

def get_input_to_network(img, input_dim=320):
    img = cv2.resize(img, (input_dim, input_dim), interpolation = cv2.INTER_CUBIC)
    img_ =  img[:,:,::-1].copy()  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = tf.convert_to_tensor(img_, dtype=tf.float32)
    return img_

# img = cv2.imread('person.jpg')

# img = cv2.resize(img, (320, 320), interpolation = cv2.INTER_CUBIC)

interpreter = tf.lite.Interpreter(model_path='final.tflite')


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(output_details)
interpreter.allocate_tensors()

img = get_input_to_network(cv2.imread('person.jpg'))
print(img.shape)
print(img.dtype)
print(input_details[0]['shape'])
# img = np.random.random(input_details[0]['shape']).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index'])
scores = interpreter.get_tensor(output_details[1]['index'])
classes = interpreter.get_tensor(output_details[2]['index'])
nums = interpreter.get_tensor(output_details[3]['index'])

print(boxes)
print(scores)
print(classes)
print(nums)
