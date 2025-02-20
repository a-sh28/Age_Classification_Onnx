
import os
import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as T
import torch


class ONNXHelper:
    def __init__(self, model_path, resolution=224):
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

    def decode_prediction(self, age_prediction):
        age = age_prediction.item()  # Convert tensor output to a scalar
        label = "child" if age <= 22 else "adult"
        return {"predicted_age": age, "label": label}

    def postprocess(self, outputs):
        return [self.postprocess_single(out) for out in outputs]

    def postprocess_single(self, output):
        return self.decode_prediction(output[0][0])

    def predict(self, image_path):
        input_data = self.preprocess_single(image_path)
        output = self.session.run(None, {"input": input_data})
        return self.postprocess_single(output[0])

    def predict_dir(self, input_dir):
        outputs = []
        for inp in self.find_images_in_dir(input_dir):
            outputs.append(self.predict(inp))
        return outputs
