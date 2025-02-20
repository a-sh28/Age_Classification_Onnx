# Age Classification ( ONNX model)

## Overview
This repository contains an implementation of an age classification model using ONNX. The model is deployed as an ONNX server integrated with the RescueBox application to predict age categories (child or adult) based on images.

---

## Clone the Repository
To get started, clone the repository:

```bash
git clone https://github.com/UMass-Rescue/age_classification.git
cd age_classification
```

---

## Setup and Install Dependencies
Ensure you have Python installed (preferably Python 3.8+). Then install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the ONNX Server on RescueBox

To start the ONNX server, follow these steps:

1. Navigate to the repository directory.
2. Run the following command:

```bash
python server.py --port 5000
```

By default, the server runs on port `5000`. You can change the port using the `--port` argument.



## Preprocessing & Postprocessing

### Preprocessing Steps:
1. Load the image and convert it to RGB.
2. Resize the image to `224x224` pixels.
3. Apply center cropping and normalize the tensor.
4. Convert to ONNX input format (numpy array of shape `(1, 3, 224, 224)`).

### Postprocessing Steps:
1. Decode the model output.
2. Convert the age prediction to categorical labels (`child` or `adult`).
3. Return the final classification result with confidence scores.

---

## Request Flow: From Input to Output

1. **User uploads images to RescueBox.**
2. **Images are sent to the ONNX based Flask ML server.**
3. **Preprocessing is applied to images.**
4. **ONNX model processes the input and returns raw predictions.**
5. **Postprocessing converts predictions into readable labels.**
6. **Predictions are saved and returned as a CSV file.**

---





