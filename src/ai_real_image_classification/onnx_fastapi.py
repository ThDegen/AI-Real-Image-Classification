import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from torchvision import transforms
from contextlib import asynccontextmanager

MODEL_PATH = "./models/resnet18.onnx"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model...")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    app.state.session = ort.InferenceSession(MODEL_PATH, providers=providers)
    app.state.input_name = app.state.session.get_inputs()[0].name
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


@app.post("/predict")
async def predict(data: UploadFile = File(...)):
    image = Image.open(data.file).convert("RGB")
    img = transform(image).unsqueeze(0).numpy()

    session = app.state.session
    input_name = app.state.input_name

    outputs = session.run(None, {input_name: img})
    logits = outputs[0]

    probability = float(1 / (1 + np.exp(-logits[0][0])))
    prediction = int(probability > 0.5)

    return {
        "probability": probability,
        "prediction": "Real Image" if prediction == 0 else "Fake Image",
    }
