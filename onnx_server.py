import argparse
import csv
import warnings
from typing import TypedDict
from pathlib import Path
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)

from onnx_helper import ONNXHelper  # Import the ONNX helper
import torch

warnings.filterwarnings("ignore")


# Configure UI Elements in RescueBox Desktop
def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to the output file",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])


# Specify the input and output types for the task
class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput


class Parameters(TypedDict):
    pass


# Create a server instance
server = MLServer(__name__)

server.add_app_metadata(
    name="Age Classification",
    author="Umass Rescue",
    version="0.1.0",
    info=load_file_as_string("img-app-info.md"),
)

# Initialize the ONNX model helper

model_helper = ONNXHelper(model_path="/Users/aravind/Desktop/age_classification/model.onnx")

@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path  # Input directory path
    output_dir = Path(inputs["output_file"].path)  # Output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir/"detections.csv"
    print(parameters)

    image_paths = model_helper.find_images_in_dir(input_path)
    saved_images = []
    for idx, img_path in enumerate(image_paths):
        saved_images.append(model_helper.predict(img_path))
    
    with open(csv_file, mode ="w+", newline= "") as f:
        writer = csv.writer(f)
        for i, result in enumerate(saved_images):
            writer.writerow(["Image_{}".format(i+1)])
            writer.writerow(["Bounding Box", "Confidence" , "Class_ID"])
            for j in result:
                writer.writerow([j["bounding_box"],j["confidence"],j["class_ids"]])
            writer.writerow([])
    return FileResponse(file_type="csv", path=str(csv_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument("--port", type=int, help="Port number to run the server", default=5000)
    args = parser.parse_args()

    print("CUDA is available." if torch.cuda.is_available() else "CUDA is not available.")
    server.run(port=args.port)
