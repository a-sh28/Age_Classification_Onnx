import torch
import onnx
import onnxruntime
from mivolo.predictor import Predictor  


model = Predictor()  
model.load_state_dict(torch.load("mivolo_model.py", map_location=torch.device('cpu')))
model.eval()  

dummy_input = torch.randn(1, 3, 224, 224)  

onnx_path = "model.onnx"
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path, 
    export_params=True, 
    opset_version=11, 
    do_constant_folding=True, 
    input_names=["input"], 
    output_names=["output"]
)

print(f"Model has been converted to ONNX format and saved as {onnx_path}")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
