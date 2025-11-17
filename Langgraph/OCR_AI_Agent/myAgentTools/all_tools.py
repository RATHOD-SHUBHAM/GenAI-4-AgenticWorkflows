import cv2
import base64
import numpy as np
import tempfile
import os

from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain.tools import tool
# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI


from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

v_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version='2025-03-01-preview'
    )

@tool
def extract_text(img_path: str) -> str:
    """
    Extracts text from an image specified by its file path. This function reads the
    image file, encodes the image data as base64, sends the image content to a
    vision-capable language model to extract text, and returns the resulting text
    content.

    :param img_path: The file path of the image from which text is to be extracted.
    :type img_path: str
    :return: Extracted text from the image, or an empty string if an error occurs
        during the process.
    :rtype: str
    :raises Exception: If any error occurs during image reading, encoding,
        or processing the response from the model.
    """

    all_text = ""

    try:
        # read image and encode as base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()
        
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # prepare the prompt including the base64 image data
        message = [
            HumanMessage(
                content = [
                    {
                        "type" : "text",
                        "text" : (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # call the vision-capable model
        response = v_llm.invoke(message)

        # append extracted text
        all_text += response.content + "\n\n"

        return all_text.strip()
    
    except Exception as e:
        # Handle error message gracefully
        error_msg = f"Error Extracting text: {str(e)}"
        print("Exception Occured: ", error_msg)

        return ""


# Upscale Image
@tool
def preprocess_image(img_path: str, op: str = "threshold", target_width: int = 1600) -> str:
    """
    Preprocesses an input image to improve text recognition or downstream OCR
    accuracy. Depending on the selected operation, the function applies
    thresholding-based binarization or deskewing, optionally upscales the image
    to a target width, and then writes the processed result to a temporary PNG
    file.

    This function supports two operations:

        - "threshold": Converts the image to grayscale, applies bilateral
          denoising, and performs adaptive thresholding to highlight text.

        - "deskew": Estimates the skew angle of the text using Otsu-thresholded
          binary data, corrects rotation, and produces a deskewed version of the
          original image.

    The output is always saved as a 3-channel PNG to ensure broad compatibility
    with vision models.

    :param img_path: Path to the input image to preprocess.
    :type img_path: str

    :param op: Image preprocessing operation to perform. Supported values are
        "threshold" (default) and "deskew".
    :type op: str

    :param target_width: Minimum width to upscale the image to, preserving aspect
        ratio. If the image is already wider or if None is provided, no scaling
        occurs. Default is 1600.
    :type target_width: int

    :return: The file path of the processed image saved as a PNG.
    :rtype: str

    :raises ValueError: If the input image cannot be read.
    :raises Exception: For unexpected processing errors such as failures in
        resizing, thresholding, or file writing.
    """
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError(f"Could not read the image from {img_path}")

    # upscale image to help OCR
    if target_width is not None and img.shape[1] < target_width:
        scale = target_width / img.shape[1]
        img = cv2.resize(img, None, fx = scale, interpolation = cv2.INTER_CUBIC)
    
    if op == "threshold":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gentle denoise and local binarisation to sharpen text
        gray = cv2.bilateralFilter(gray, 7, 50, 50)
        out = cv2.adaptiveThreshold(gray,
                                    255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,
                                    31,
                                    10)

    elif op == "deskew":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        
        # Otsu to find text pixels
        _, bw = cv2.threshold(gray,
                              0,
                              255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(bw > 0))
        angle = 0.0
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            # Convert OpenCV's angle convention to a proper rotation
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        # Rotate around centre
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Ensure 3-channel PNG for the vision model (it accepts grayscale too, but PNG-3 is universal)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Write to a temporary PNG and return its path
    tmpdir = tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(tmpdir, f"{base}_proc_{op}.png")
    cv2.imwrite(out_path, out)
    return out_path


# Temperature
@tool
def convert_temperature_fahrenheit_to_celsius(fahrenheit_temperature: int) -> int:
    """
    Convert a given temperature from Fahrenheit to Celsius.

    This function takes a temperature value in Fahrenheit and converts it into
    its equivalent in Celsius using the standard formula. The output temperature
    is rounded to the nearest integer.

    :param fahrenheit_temperature: The temperature in Fahrenheit to be converted.
    :type fahrenheit_temperature: int
    :return: The equivalent temperature in Celsius, rounded to the nearest integer.
    :rtype: int
    """
    return round((fahrenheit_temperature - 32) * 5 / 9)


# Unit Conversions
@tool
def convert_length_inches_to_cm(inches_length: int) -> int:
    """
    Converts a given length from inches to centimeters.

    This function takes a length measurement in inches and converts it to centimeters
    using the conversion factor of 2.54. The resulting value is rounded to the nearest
    integer and returned.

    :param inches_length: The length in inches to be converted
    :type inches_length: int
    :return: The length converted to centimeters, rounded to the nearest integer
    :rtype: int
    """
    return round(inches_length * 2.54)

@tool
def convert_weight_cups_to_grams(cups_weight: float) -> int:
    """
    Converts weight from cups to grams. This function accepts a weight in cups and
    returns the equivalent weight in grams. It assumes a standard conversion
    rate of 1 cup = 250 grams.

    :param cups_weight: The weight in cups
    :type cups_weight: float
    :return: The equivalent weight in grams
    :rtype: int
    """
    return round(cups_weight * 250)

@tool
def convert_volume_cups_to_millilitres(cups_volume: float) -> int:
    """
    This function takes a floating-point value representing the volume in cups
    and converts it into millilitres by multiplying with a conversion factor of 240.
    The result is rounded to the nearest integer to ensure accurate representation
    of the millilitres value.

    :param cups_volume: The volume in cups to be converted.
    :type cups_volume: float
    :return: The equivalent volume in ml.
    :rtype: int
    """
    return round(cups_volume * 240)