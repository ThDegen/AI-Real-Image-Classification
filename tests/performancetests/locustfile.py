import io
from PIL import Image
from locust import HttpUser, task, between

class ImagePredictionUser(HttpUser):
    wait_time = between(1, 3) # simulate thinking time

    def on_start(self):
        """
        Generate a dummy image in memory once when the user starts
        """
        self.image_data = self._generate_dummy_image()

    def _generate_dummy_image(self):
        img = Image.new('RGB', (224, 224), color='red')
        
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr.read()

    @task
    def predict(self):
        """
        Send a POST request to the prediction endpoint with the image
        """
        files = {
            'data': ('test_image.jpg', self.image_data, 'image/jpeg')
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200: # verify if we get a 200 OK response
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}: {response.text}")
