import numpy as np
import onnxruntime as ort
import time
import json
import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from PIL import Image
from contextlib import asynccontextmanager
from google.cloud import storage
from prometheus_client import Counter, Histogram, Summary, make_asgi_app

# --- Configuration ---
MODEL_PATH = "./models/resnet18.onnx"
BUCKET_NAME = "ai-real-image-classification-storage"
CLASS_NAMES = ["Real Image", "Fake Image"]

# Metrics
request_counter = Counter("prediction_requests_total", "Total number of API requests")
error_counter = Counter("prediction_errors_total", "Total number of prediction errors")
prediction_latency = Histogram(
    "prediction_duration_seconds", "Time taken for prediction"
)
image_size_summary = Summary("image_size_bytes", "Size of the uploaded images in bytes")


def preprocess_numpy(image: Image.Image):
    image = image.resize((224, 224), resample=Image.BILINEAR)
    img_data = np.array(image).astype("float32") / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))

    return img_data[np.newaxis, :]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading ONNX model...")
    # Restrict threads to save memory in Cloud Run
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    providers = ["CPUExecutionProvider"]  # Standard for Cloud Run
    app.state.session = ort.InferenceSession(
        MODEL_PATH, sess_options=opts, providers=providers
    )
    app.state.input_name = app.state.session.get_inputs()[0].name
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


def save_prediction_to_gcp(filename: str, probability: float, prediction: str):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        timestamp = datetime.datetime.now(tz=datetime.UTC)
        data = {
            "filename": filename,
            "probability": probability,
            "prediction": prediction,
            "timestamp": timestamp.isoformat(),
        }
        blob_name = f"logs/prediction_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.json"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(json.dumps(data), content_type="application/json")
    except Exception as e:
        print(f"GCS Log Error: {e}")


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, data: UploadFile = File(...)):
    request_counter.inc()
    start_time = time.perf_counter()

    try:
        # Observe file size
        file_bytes = await data.read()
        image_size_summary.observe(len(file_bytes))
        await data.seek(0)

        image = Image.open(data.file).convert("RGB")
        img = preprocess_numpy(image)

        # Inference
        session = app.state.session
        input_name = app.state.input_name
        outputs = session.run(None, {input_name: img})
        logits = outputs[0]

        # Post-processing
        probability = float(1 / (1 + np.exp(-logits[0][0])))
        prediction_idx = int(probability > 0.5)
        prediction_label = CLASS_NAMES[prediction_idx]

        prediction_latency.observe(time.perf_counter() - start_time)

        background_tasks.add_task(
            save_prediction_to_gcp, data.filename, probability, prediction_label
        )

        return {"probability": probability, "prediction": prediction_label}

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))
