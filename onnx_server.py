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
from onnx_helper import ONNXModelHelper  # Import the ONNX helper
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
model_helper = ONNXModelHelper(model_path="model.onnx")


@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path  # Input directory path
    output_dir = Path(inputs["output_file"].path)  # Output directory
    output_file = str(output_dir / f"predictions_{int(torch.rand(1) * 1000)}.csv")

    print(parameters)

    # Get predictions for all images in the directory
    image_paths = model_helper.find_images_in_dir(input_path)
    predictions = [model_helper.predict(img_path) for img_path in image_paths]

    # Write results to CSV file
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "prediction", "confidence"]
        )
        writer.writeheader()  # Write header row
        for img_path, pred in zip(image_paths, predictions):
            writer.writerow({
                "image_path": img_path,
                "prediction": pred["prediction"],  # "child" or "adult"
                "confidence": pred["confidence"]
            })

    return ResponseBody(FileResponse(path=output_file, file_type="csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument("--port", type=int, help="Port number to run the server", default=5000)
    args = parser.parse_args()

    print("CUDA is available." if torch.cuda.is_available() else "CUDA is not available.")
    server.run(port=args.port)
