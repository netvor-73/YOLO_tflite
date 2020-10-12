from silence_tensorflow import silence_tensorflow

silence_tensorflow()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32)
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

# np.array([(10, 13), (16, 30), (33, 23)])
# np.array([(30, 61), (62, 45), (59, 119)])
# np.array([(116, 90), (156, 198), (373, 326)])

YOLOV3_LAYER_LIST = [
    'Darknet_53',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]


def darknetConv(x, out_filters, size, strides=1, batch_norm=True, use_bias=False):
    if strides == 1:
        padding = 'same'
    else:
        x = layers.ZeroPadding2D(padding=1)(x)
        padding = 'valid'

    x = layers.Conv2D(out_filters, kernel_size=size, strides=strides,
                        padding=padding, use_bias=use_bias)(x)

    if batch_norm:
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.LeakyReLU(alpha=0.1)(x)

    return x

def darknetResidual(x, filters):
    prev = x
    x = darknetConv(x, filters // 2, 1)
    x = darknetConv(x, filters, 3)
    x = layers.Add()([prev, x])

    return x

def darknetBlock(x, filters, block):
    # x = darknetConv(x, filters, 3, strides=2)
    for _ in range(block):
        x = darknetResidual(x, filters)
    return x

def darknet53(name=None):
    x = input = keras.Input([None, None, 3])
    x = darknetConv(x, 32, 3)
    x = darknetConv(x, 64, 3, strides=2)
    x = darknetBlock(x, 64, 1)
    x = darknetConv(x, 128, 3, strides=2)
    x = darknetBlock(x, 128, 2)
    x = darknetConv(x, 256, 3, strides=2)
    x = x_36 = darknetBlock(x, 256, 8)
    x = darknetConv(x, 512, 3, strides=2)
    x = x_61 = darknetBlock(x, 512, 8)
    x = darknetConv(x, 1024, 3, strides=2)
    x = darknetBlock(x, 1024, 4)

    return keras.Model(input, (x_36, x_61, x), name=name)

def YoloConv(filters, name='None'):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = keras.Input(x_in[0].shape[1:]), keras.Input(x_in[1].shape[1:])
            x, x_route = inputs

            x = darknetConv(x, filters, 1)
            # x = layers.UpSampling2D(size=2, interpolation='bilinear')(x)
            x = tf.image.resize(x, [tf.shape(x)[1] * 2, tf.shape(x)[1] * 2])
            x = layers.Concatenate()([x, x_route])

        else:
            x = inputs = keras.Input(x_in.shape[1:])

        x = darknetConv(x, filters, 1)
        x = darknetConv(x, filters * 2, 3)
        x = darknetConv(x, filters, 1)
        x = darknetConv(x, filters * 2, 3)
        x = darknetConv(x, filters, 1)

        return keras.Model(inputs, x, name=name)(x_in)

    return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = keras.Input(x_in.shape[1:])
        x = darknetConv(x, filters * 2, 3)
        x = darknetConv(x, anchors * (classes + 5), 1, batch_norm=False, use_bias=True)
        # x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            # anchors, classes + 5)))(x)
        # x = tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5))
        return keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def yolo_boxes(output, anchors, classes, shape):

    # output = tf.make_ndarray(output)

    # output = tf.reshape(output, (-1, tf.shape(output)[1], tf.shape(output)[2], len(anchors), classes + 5))

    output = tf.reshape(output, (1, shape, shape, 3, 85))

    # print(output.shape)

    grid_size = tf.shape(output)[1]
    # stride = 416 / tf.shape(output)[1]
    box_xy, box_wh, objectness, class_probs = tf.split(output, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)

    # grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    # grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    y = tf.tile(tf.range(grid_size, dtype=tf.int32)[:, tf.newaxis], [1, grid_size])
    x = tf.tile(tf.range(grid_size, dtype=tf.int32)[tf.newaxis, :], [grid_size, 1])

    xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [tf.shape(output)[0], 1, 1, 3, 1])
    # xy_grid = tf.cast(xy_grid, tf.float32)

    box_xy = (box_xy + tf.cast(xy_grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs

# class YoloNms(keras.layers.Layer):
#     def __init__(self, name):
#         super(YoloNms, self).__init__(name=name)
#
#     def call(self, inputs):
#
#         bbox = tf.concat([
#                     tf.reshape(inputs[0][0], (tf.shape(inputs[0][0])[0], 300, tf.shape(inputs[0][0])[-1])),
#                     tf.reshape(inputs[1][0], (tf.shape(inputs[1][0])[0], 1200, tf.shape(inputs[1][0])[-1])),
#                     tf.reshape(inputs[2][0], (tf.shape(inputs[2][0])[0], 4800, tf.shape(inputs[2][0])[-1]))
#         ], axis=1)
#
#         confidence = tf.concat([
#                     tf.reshape(inputs[0][1], (tf.shape(inputs[0][1])[0], 300, tf.shape(inputs[0][1])[-1])),
#                     tf.reshape(inputs[1][1], (tf.shape(inputs[1][1])[0], 1200, tf.shape(inputs[1][1])[-1])),
#                     tf.reshape(inputs[2][1], (tf.shape(inputs[2][1])[0], 4800, tf.shape(inputs[2][1])[-1]))
#         ], axis=1)
#
#         class_probs = tf.concat([
#                     tf.reshape(inputs[0][2], (tf.shape(inputs[0][2])[0], 300, tf.shape(inputs[0][2])[-1])),
#                     tf.reshape(inputs[1][2], (tf.shape(inputs[1][2])[0], 1200, tf.shape(inputs[1][2])[-1])),
#                     tf.reshape(inputs[2][2], (tf.shape(inputs[2][2])[0], 4800, tf.shape(inputs[2][2])[-1]))
#         ], axis=1)
#
#
#         scores = confidence * class_probs
#
#         box_classes = tf.argmax(scores, axis=-1)
#         class_scores = keras.backend.max(scores, axis=-1)
#         #
#         filtering_mask = class_scores >= .6
#         # #
#         scores = tf.boolean_mask(class_scores, filtering_mask)
#         boxes = tf.boolean_mask(bbox, filtering_mask)
#         classes = tf.boolean_mask(box_classes, filtering_mask)
#
#         nms_indices_, nums = tf.image.non_max_suppression_padded(boxes, scores, 10,
#                                                       score_threshold=0.6,
#                                                       iou_threshold=0.5,
#                                                       pad_to_max_output_size=True)
#         nums = tf.expand_dims(nums, axis=-1)
#         # print('I am here', nms_indices)
#         nms_indices = tf.slice(nms_indices_, tf.constant([0], dtype=tf.int32), nums)
#
#         # nms_indices = tf.image.non_max_suppression(boxes, scores, 10)
#         scores = tf.expand_dims(tf.gather(scores, nms_indices), axis=0)
#         boxes = tf.expand_dims(tf.gather(boxes, nms_indices), axis=0)
#         classes = tf.expand_dims(tf.gather(classes, nms_indices), axis=0)
#
#         return boxes, scores, classes, nums

def yolo_nms(inputs, anchors, masks, classes):
    # boxes, conf, type
    # b, c, t = [], [], []

    # for o in outputs:
    #     b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
    #     c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
    #     t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    # bbox = tf.concat(b, axis=1)
    # confidence = tf.concat(c, axis=1)
    # class_probs = tf.concat(t, axis=1)


    bbox = tf.concat([
                tf.reshape(inputs[0][0], (tf.shape(inputs[0][0])[0], 300, tf.shape(inputs[0][0])[-1])),
                tf.reshape(inputs[1][0], (tf.shape(inputs[1][0])[0], 1200, tf.shape(inputs[1][0])[-1])),
                tf.reshape(inputs[2][0], (tf.shape(inputs[2][0])[0], 4800, tf.shape(inputs[2][0])[-1]))
    ], axis=1)

    confidence = tf.concat([
                tf.reshape(inputs[0][1], (tf.shape(inputs[0][1])[0], 300, tf.shape(inputs[0][1])[-1])),
                tf.reshape(inputs[1][1], (tf.shape(inputs[1][1])[0], 1200, tf.shape(inputs[1][1])[-1])),
                tf.reshape(inputs[2][1], (tf.shape(inputs[2][1])[0], 4800, tf.shape(inputs[2][1])[-1]))
    ], axis=1)

    class_probs = tf.concat([
                tf.reshape(inputs[0][2], (tf.shape(inputs[0][2])[0], 300, tf.shape(inputs[0][2])[-1])),
                tf.reshape(inputs[1][2], (tf.shape(inputs[1][2])[0], 1200, tf.shape(inputs[1][2])[-1])),
                tf.reshape(inputs[2][2], (tf.shape(inputs[2][2])[0], 4800, tf.shape(inputs[2][2])[-1]))
    ], axis=1)

    scores = confidence * class_probs

    box_classes = tf.argmax(scores, axis=-1)
    class_scores = keras.backend.max(scores, axis=-1)
    #
    filtering_mask = class_scores >= .6
    # #
    scores = tf.boolean_mask(class_scores, filtering_mask)
    boxes = tf.boolean_mask(bbox, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    nms_indices_, nums = tf.image.non_max_suppression_padded(boxes, scores, 10,
                                                  score_threshold=0.6,
                                                  iou_threshold=0.5,
                                                  pad_to_max_output_size=True)
    nums = tf.expand_dims(nums, axis=-1)
    # print('I am here', nms_indices)
    # nms_indices = tf.slice(nms_indices_, tf.constant([0], dtype=tf.int32), nums)

    # nms_indices = tf.image.non_max_suppression(boxes, scores, 10)
    scores = tf.expand_dims(tf.gather(scores, nms_indices_), axis=0)
    boxes = tf.expand_dims(tf.gather(boxes, nms_indices_), axis=0)
    classes = tf.expand_dims(tf.gather(classes, nms_indices_), axis=0)

    return tf.cast(boxes, dtype=tf.float32), tf.cast(scores, dtype=tf.float32), tf.cast(classes, tf.float32), tf.cast(nums, tf.float32)


def YOLOv3(size=320, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks,
            classes=80):

    x = input = keras.Input([size, size, channels], name='input')

    x_36, x_61, x = darknet53(name='Darknet_53')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)


    boxes_0 = layers.Lambda(lambda x: yolo_boxes(x, np.array([(116, 90), (156, 198), (373, 326)])/320, classes, shape=10), name='yolo_boxes_0')(output_0)
    boxes_1 = layers.Lambda(lambda x: yolo_boxes(x, np.array([(30, 61), (62, 45), (59, 119)])/320, classes, shape=20), name='yolo_boxes_1')(output_1)
    boxes_2 = layers.Lambda(lambda x: yolo_boxes(x, np.array([(10, 13), (16, 30), (33, 23)])/320, classes, shape=40), name='yolo_boxes_2')(output_2)

    outputs = layers.Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0, boxes_1, boxes_2))

    # outputs = YoloNms(name='nms')((boxes_0, boxes_1, boxes_2))

    return keras.Model(input, outputs, name='yolov3')


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)


    layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
