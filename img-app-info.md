# Age Classification ( ONNX model)

The pretrained model used in this project was taken from the [UMass Rescue Age Classification repository](https://github.com/UMass-Rescue/age_classification).


## Overview
This repository contains an implementation of an age - gender classification model using ONNX. The model is deployed as an ONNX server integrated with the RescueBox application to detect portions of an image and draw bounding boxes over it ( which can later be used for gender/age prediction).

---

## Clone the Repository
To get started, clone the repository:

```bash
git clone https://github.com/a-sh28/Age_Classification_Onnx.git
cd Age_Classification_Onnx
```

---

## Setup and Install Dependencies
Ensure you have Python installed (Python 3.10+ for flask_ML). Then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the ONNX Server on RescueBox

To start the ONNX server, follow these steps:

1. Navigate to the repository directory.
2. Run the following command:

```bash
python onnx_server.py --port 5000
```

By default, the server runs on port `5000`. You can change the port using the `--port` argument.


### Exporting the Model to ONNX

The pretrained model was exported from PyTorch to ONNX format by first loading the model and creating a dummy input tensor. Then, the model was exported using `torch.onnx.export()` with appropriate settings for export parameters and input/output names. After exporting, the ONNX model was validated using the `onnx.checker.check_model()` function to ensure it was correct and ready for use.


## Preprocessing & Postprocessing

### Preprocessing Steps:
1. Load the image and convert it to RGB.
2. Resize the image to `640x640` pixels.
3. Apply center cropping and normalize the tensor.
4. Convert to ONNX input format (numpy array of shape `(1, 3, 640, 640)`).

### Postprocessing Steps:
1. Decode the model output.
2. Convert the detections of bounding boxes, confidence and class_ids from raw format into csv.
3. Return the detections as a csv file.

---

## Request Flow: From Input to Output

1. **User uploads images to RescueBox.**
2. **Images are sent to the ONNX based Flask ML server.**
3. **Preprocessing is applied to images.**
4. **ONNX model processes the input and returns raw predictions.**
5. **Postprocessing converts predictions into readable labels.**
6. **Predictions are saved and returned as a CSV file.**

---





