import base64
from mimetypes import guess_type
from PIL import Image
from langchain_nvidia_ai_endpoints import ChatNVIDIA

class ImageAnalyst:
    def __init__(self):
        self.client = ChatNVIDIA(model="nvidia/neva-22b",
                                 api_key='YOUR_NVIDIA_API_KEY',)

    def resize_image(self, image_path, max_size = (800, 800)):
        resized_img_path = 'resized_image.jpg'
        image = Image.open(image_path)
        resized_image = image.resize(max_size)
        resized_image.save(resized_img_path)
        return resized_img_path

    # Function to encode a local image into data URL
    def local_image_to_data_url(self, image_path):
        # Guess the MIME type of the image based on the file extension
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'  # Default MIME type if none is found

        # Read and encode the image file
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def analyseImage(self, data_url):
        system_prompt = """
            You are a safety monitoring agent within a simulated factory environment on a busy factory floor where workers are engaged in various tasks.
            Your primary responsibility is to monitor the environment closely for any hazardous incidents.
            You Scan the environment for potential hazards, paying close attention to workers' movements, machinery operations, and the stability of objects on racks.
            If you detect any hazardous incidents, such as a worker falling, boxes or objects falling from racks, or machinery malfunctioning, you must identify that and log that incident in a manner that allows further communication.
        """

        response = self.client.invoke(
            [
                {"role": "system",
                 "content": system_prompt
                 },
                {"role": "user",
                 "content": [
                     {
                         "type": "text",
                         "text": "Describe this image: "
                     },
                     {
                         "type": "image_url",
                         "image_url":
                             {
                                 "url": data_url
                             }
                     }
                 ]}
            ],
            max_tokens=1024
        )

        return response.content

    def pipe(self, image_path):
        resized_image_path = self.resize_image(image_path)
        data_url = self.local_image_to_data_url(resized_image_path)
        response = self.analyseImage(data_url=data_url)
        return response



if __name__ == '__main__':
    obj = ImageAnalyst()
    print(obj.pipe(image_path = '/Users/shubhamrathod/PycharmProjects/ThirdeyeNIM/factory/test_1.jpg'))