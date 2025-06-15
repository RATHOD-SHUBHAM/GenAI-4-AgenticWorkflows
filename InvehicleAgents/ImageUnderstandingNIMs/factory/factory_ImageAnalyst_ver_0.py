import os
import requests, base64
from PIL import Image

HOME = os.getcwd()

invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"
stream = False

class ImageAnalyst():
    def resize_image(self, image_path, max_size=(800, 800)):
        resized_img_path = f'{HOME}/resized_image.jpg'
        image = Image.open(image_path)
        resized_image = image.resize(max_size)
        resized_image.save(resized_img_path)
        return resized_img_path

    def runAnalysis(self, resize_image_path):

        with open(resize_image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()

        assert len(image_b64) < 180_000, \
            "To upload larger images, use the assets API (see docs)"

        headers = {
            "Authorization": "Bearer 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",
            "Accept": "application/json"
        }

        prompt = """
            Your primary task is to closely check the surroundings for any short circuits or sparkling light bulbs.
            
            Your Task:
            Explain the incident seen in the image alongside providing some context on when and how the incident happened.
        """


        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} <img src="data:image/jpg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.20,
            "top_p": 0.70,
            "seed": 0,
            "stream": stream
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            for line in response.iter_lines():
                if line:
                    print(line.decode("utf-8"))
        else:
            result = response.json()
            # print(result['choices'][0]['message']['content'])
            return result['choices'][0]['message']['content']

    def startAnalysis(self, image_path):
        resize_image_path = self.resize_image(image_path = image_path)
        response = self.runAnalysis(resize_image_path = resize_image_path)
        return response






