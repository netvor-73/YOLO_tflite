from yolo_keras import YOLOv3, load_darknet_weights
import cv2
import numpy as np
import tensorflow as tf
import pathlib

yolo = YOLOv3(classes=80)
#
load_darknet_weights(yolo, 'yolov3.weights')
#
# # img = np.random.random((1, 320, 320, 3)).astype(np.float32)
# #
# # output = yolo(img)
#
tf.saved_model.save(yolo, './1')
#
# yolo.save('model.h5')
converter = tf.lite.TFLiteConverter.from_saved_model('./1')
#

# model = tf.keras.models.load_model('model.h5', compile=False)
# input_shape = model.inputs[0].shape.as_list()


# print(input_shape)

# input_shape[0] = 1
# func = tf.function(model).get_concrete_function(
    # tf.TensorSpec(input_shape, model.inputs[0].dtype))

# converter = tf.lite.TFLiteConverter.from_concrete_functions([func])

converter.optimizations = [tf.lite.Optimize.DEFAULT]
# sess = tf.keras.backend.get_session()

# converter = tf.lite.TFLiteConverter.from_session(sess, model.inputs, model.outputs)

# converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

# converter.experimental_new_converter = True

tflite_model = converter.convert()

tflite_model_file = pathlib.Path('model.tflite')
tflite_model_file.write_bytes(tflite_model)
