import torch
import torchvision
from model import Darknet

dummy_input = torch.randn(1, 3, 416, 416)
model = Darknet('yolov3.cfg', False)
model.load_weights('yolov3.weights')

torch.onnx.export(model, dummy_input, 'model.onnx')
