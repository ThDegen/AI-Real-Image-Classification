from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch
from PIL import Image
import io

# import app
from src.ai_real_image_classification.api import app

# create test client
client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == 200

@patch("src.ai_real_image_classification.api.model")
def test_predict_real_image(mock_model):
    # mock because model would take time to load
    mock_output = torch.tensor([[-1.0]]) 
    mock_model.return_value = mock_output
    
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = iter([mock_param])

    # dummy image in memory
    img = Image.new('RGB', (100, 100), color='black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    # send request
    response = client.post(
        "/predict/",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    )

    # assertion
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["prediction"] == "Real"
    assert "probability" in json_response

@patch("src.ai_real_image_classification.api.model")
def test_predict_invalid_input(mock_model):
    response = client.post(
        "/predict/",
        files={"file": ("test.txt", b"this is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "File provided is not an image" in response.json()["detail"]