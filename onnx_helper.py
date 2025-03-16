
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
import torch


class ONNXHelper:

    def __init__(self, model_path, resolution=640):
        self.resolution = resolution
        self.model_path = model_path
        self.valid_extensions = (".jpg", ".jpeg", ".png")
        self.session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

    def apply_transforms_on_path(self, path):
        image = Image.open(path).convert("RGB")
        return self.apply_transforms(image)

    def apply_transforms(self, image):
        transform = T.Compose([
            T.Resize(self.resolution + self.resolution // 8, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(self.resolution),
            T.ToTensor(),
        ])
        return transform(image)[None, :].numpy()

    def find_images_in_dir(self, directory):
        return [
            os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(self.valid_extensions)
        ]

    def preprocess(self, dir_path):
        return [self.preprocess_single(path) for path in self.find_images_in_dir(dir_path)]

    def preprocess_single(self, image_path):
        return self.apply_transforms_on_path(image_path)

    def decode_prediction(self, output):
        output = output.squeeze(0)
        x_center,y_center,width,height,confidence,class_ids = output
        mask = confidence > 0.6
        x_center = x_center[mask]
        y_center = y_center[mask]
        width = width[mask]
        height = height[mask]
        confidence = confidence[mask]
        class_ids = class_ids[mask]
        x1 = x_center - width/2
        x2 = x_center + width/2
        y1= y_center - height/2
        y2 = y_center + height/2
        class_ids = [int(class_id) for class_id in class_ids]
        detection_boxes = []
        for i in range(len(x1)):
            detection_boxes.append({"bounding_box":[x1[i],x2[i],y1[i],y2[i]],"confidence":confidence[i],"class_ids":class_ids[i]})
        return detection_boxes


    def postprocess(self, outputs):
        return [self.postprocess_single(out) for out in outputs]

    def postprocess_single(self, output):
        return self.decode_prediction(output)


    def predict(self, image_path):
        input_data = self.preprocess_single(image_path)
        output = self.session.run(None, {"input": input_data})
        return self.postprocess_single(output[0])

    def predict_dir(self, input_dir):
        outputs = []
        for inp in self.find_images_in_dir(input_dir):
            outputs.append(self.predict(inp))
        return outputs
