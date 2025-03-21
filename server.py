import argparse
import os
import json
import csv
from typing import TypedDict

import cv2
import torch
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (DirectoryInput,
                                             EnumParameterDescriptor, EnumVal,
                                             FileResponse, InputSchema,
                                             InputType, ParameterSchema,
                                             ResponseBody, TaskSchema,
                                             TextParameterDescriptor,
                                             BatchFileResponse)
from mivolo.predictor import Predictor

def get_images(folder_dir):
    images = []
    for image in os.listdir(folder_dir):
        if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"):
            images.append(os.path.join(folder_dir, image))
    return images

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-weights", default="models/yolov8x_person_face.pt", type=str)
    parser.add_argument("--checkpoint", default="models/mivolo_imdb.pth.tar", type=str)
    parser.add_argument("--with-persons", action="store_false")
    parser.add_argument("--disable_faces", action="store_true")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--single-person", action="store_true")
    parser.add_argument("--draw", action="store_false")
    parser.add_argument("--port", default=5000, type=int)
    return parser

def generate_img_data(imgs):
    data = []
    for img in imgs:
        img_data = []
        img_data.append(img["file_path"])
        nf = nm = nc = na = 0
        for person in img["result"]:
            if person["label"] == "child": nc += 1
            else: na += 1
            if person["gender"] == "male": nm += 1
            else: nf += 1
        img_data.append(nf)
        img_data.append(nm)
        img_data.append(nc)
        img_data.append(na)
        data.append(img_data)
    return data

def classify_given_age(age):
    return "child" if age <= 22 else "adult"

def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_directory",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_directory",
        label="Path to the output directory",
        input_type=InputType.DIRECTORY,
    )
    disable_faces_schema = ParameterSchema(
        key="single_person",
        label="Single person flag",
        value=EnumParameterDescriptor(
            default="False",
            enum_vals=[
                EnumVal(key="True", label="True"),
                EnumVal(key="False", label="False"),
            ],
        ),
    )
    store_images_schema = ParameterSchema(
        key="store_images",
        label="Store images",
        value=EnumParameterDescriptor(
            default="True",
            enum_vals=[
                EnumVal(key="True", label="True"),
                EnumVal(key="False", label="False"),
            ],
        ),
    )
    return TaskSchema(
        inputs=[input_schema, output_schema], parameters=[disable_faces_schema, store_images_schema]
    )

parser = get_parser()
params = parser.parse_args()
predictor = Predictor(params, verbose=False)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

server = MLServer(__name__)

server.add_app_metadata(
    name="Age and Gender Classifier",
    author="User",
    version="1.0.0",
    info=load_file_as_string("README.md"),
)

class Inputs(TypedDict):
    input_directory: DirectoryInput
    output_directory: DirectoryInput

class Params(TypedDict):
    single_person: str
    store_images: str


@server.route("/classify_age_gender", task_schema_func=create_transform_case_task_schema)
def classify(inputs: Inputs, parameters: Params) -> ResponseBody:
    input_folder_dir = inputs["input_directory"].path
    output_folder_dir = inputs["output_directory"].path
    hash_name = str(torch.randint(0, 1000000000, (1,)).item())
    images = get_images(input_folder_dir)
    single_person_flag = True if parameters["single_person"] == "True" else False
    store_images = True if parameters["store_images"] == "True" else False
    if store_images:
        os.makedirs(os.path.join(output_folder_dir, "outputs_" + hash_name), exist_ok=True)
    output_images_path = os.path.join(output_folder_dir, "outputs_" + hash_name)

    main_res = []
    no_predict = 0
    for image_name in images:
        avg_age = 0
        res = []
        img = cv2.imread(image_name)
        import pdb
        pdb.set_trace()
        detected_objects, output_img = predictor.recognize(img)
        if store_images:
            cv2.imwrite(os.path.join(output_images_path, os.path.basename(image_name)), output_img)
        bboxes = detected_objects.yolo_results.boxes.xyxy.cpu().numpy()
        ages = detected_objects.ages
        genders = detected_objects.genders
        face_indexes = detected_objects.face_to_person_map.keys()
        for i in face_indexes:
            if ages[i] is not None:
                avg_age += ages[i]
                res.append(
                    {
                        "bbox": {"X1": int(bboxes[i][0]), "Y1": int(bboxes[i][1]), "X2": int(bboxes[i][2]), "Y2": int(bboxes[i][3])},
                        "label": classify_given_age(int(ages[i])),
                        "gender": genders[i],
                    }
                )
        if not res:
            no_predict += 1
            main_res.append({"file_path": image_name, "result": "No person detected"})
            continue  
        if single_person_flag:
            res = [res[0]]
            res[0]["label"] = classify_given_age(int(avg_age / len(face_indexes)))
        main_res.append({"file_path": image_name, "result": res})
    
    result_path = os.path.join(output_folder_dir, hash_name + "_result.json")
    with open(result_path, "w") as f:
        json.dump(main_res, f, indent=4)

    img_data_dict = generate_img_data(main_res)
    csv_path = os.path.join(output_folder_dir, hash_name + "_csv_info.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "Num of detected females", "Num of detected males", "Num of detected children", "Num of detected adults"])
        writer.writerows(img_data_dict)
    res_body = [FileResponse(path=result_path, file_type="json"), FileResponse(path=csv_path, file_type="csv")]
    return ResponseBody(BatchFileResponse(files=res_body))

server.run(port=params.port)

