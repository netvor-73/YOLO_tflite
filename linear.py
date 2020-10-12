import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt

x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

# model = tf.keras.models.Sequential([
#         tf.keras.layers.Dense(units=1, input_shape=[1])])
#
# model.compile(optimizer='sgd', loss='mean_squared_error')
#
# model.fit(x, y, epochs=200)
#
#
export_dir = 'saved_model/1'
# tf.saved_model.save(model, export_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

# tflite_model_file = pathlib.Path('model.tflite')
# tflite_model_file.write_bytes(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

inputs, outputs = [], []

for _ in range(100):
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]['index'])

    inputs.append(input_data[0][0])
    outputs.append(np.array(tflite_results)[0][0])

plt.plot(inputs, outputs, 'r')
plt.show()
