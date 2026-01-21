from contextlib import asynccontextmanager
from http import HTTPStatus
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms
from PIL import Image
import io

from .model import Model

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    loads model into memory
    """
    global model
    print("Loading model")
    try:
        # loaf from checkpoint
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model = Model.load_from_checkpoint("models/best.ckpt", map_location=device)
        
        # eval mode
        model.eval() 
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
    
    yield
    
    print("Shutting down")

app = FastAPI(title="AI vs Real Image Classifier", lifespan=lifespan)

# image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "message": "AI vs Human Image Classification API",
        "status": HTTPStatus.OK
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict if an image is AI-generated or Real
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # read image file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # transform image
        input_tensor = transform(image)
        
        input_batch = input_tensor.unsqueeze(0)

        device = next(model.parameters()).device
        input_batch = input_batch.to(device)

        # predicton
        with torch.no_grad():
            logits = model(input_batch)
            
            # sigmoid for prob
            prob = torch.sigmoid(logits).item()
            
            # TODO: check labels, for my random test I got the wrong classification
            prediction = "Synthetic" if prob > 0.5 else "Real"

        return {
            "filename": file.filename,
            "prediction": prediction,
            "probability": prob,
            "status-code": HTTPStatus.OK
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")