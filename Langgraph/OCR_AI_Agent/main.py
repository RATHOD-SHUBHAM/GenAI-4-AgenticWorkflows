from typing_extensions import TypedDict, Optional, Annotated

from langchain.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
# Import Azure OpenAI
from langchain_openai import AzureChatOpenAI

# Langgraph
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

# Step 1: Define tools and model
from myAgentTools.all_tools import (
    extract_text,
    preprocess_image,
    convert_temperature_fahrenheit_to_celsius,
    convert_length_inches_to_cm,
    convert_weight_cups_to_grams,
    convert_volume_cups_to_millilitres
)

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version='2025-03-01-preview'
    )

tools = [
    extract_text,
    preprocess_image,
    convert_temperature_fahrenheit_to_celsius,
    convert_length_inches_to_cm,
    convert_weight_cups_to_grams,
    convert_volume_cups_to_millilitres
]

tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# 2. Define State
class MessagesState(TypedDict):
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int

# 3. Define Nodes
def assistant(state: MessagesState):
    textual_description_of_tools = """
    def extract_text(img_path: str) -> str:
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
        
    def preprocess_image(img_path: str, op: str = "threshold", target_width: int = 1600) -> str:
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
        
    def convert_temperature_fahrenheit_to_celsius(fahrenheit_temperature: int) -> int:

        Convert a given temperature from Fahrenheit to Celsius.

        This function takes a temperature value in Fahrenheit and converts it into
        its equivalent in Celsius using the standard formula. The output temperature
        is rounded to the nearest integer.

        :param fahrenheit_temperature: The temperature in Fahrenheit to be converted.
        :type fahrenheit_temperature: int
        :return: The equivalent temperature in Celsius, rounded to the nearest integer.
        :rtype: int

    def convert_length_inches_to_cm(inches_length: int) -> int:
        Converts a given length from inches to centimeters.

        This function takes a length measurement in inches and converts it to centimeters
        using the conversion factor of 2.54. The resulting value is rounded to the nearest
        integer and returned.

        :param inches_length: The length in inches to be converted
        :type inches_length: int
        :return: The length converted to centimeters, rounded to the nearest integer
        :rtype: int
    
    def convert_weight_cups_to_grams(cups_weight: float) -> int:
        Converts weight from cups to grams. This function accepts a weight in cups and
        returns the equivalent weight in grams. It assumes a standard conversion
        rate of 1 cup = 250 grams.

        :param cups_weight: The weight in cups
        :type cups_weight: float
        :return: The equivalent weight in grams
        :rtype: int
    
    def convert_volume_cups_to_millilitres(cups_volume: float) -> int:
        This function takes a floating-point value representing the volume in cups
        and converts it into millilitres by multiplying with a conversion factor of 240.
        The result is rounded to the nearest integer to ensure accurate representation
        of the millilitres value.

        :param cups_volume: The volume in cups to be converted.
        :type cups_volume: float
        :return: The equivalent volume in ml.
        :rtype: int
    """

    image = state["input_file"]
    
    sys_msg = SystemMessage(content = f"""
    You are a helpful assistant. You can analyse documents with provided tools:\n{textual_description_of_tools}.
    
    You have access to some optional images. Currently the loaded image is: {image}""")

    return {
        "messages" : [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file" : state["input_file"]
    }


"""
The ReAct Pattern:
Allow me to explain the approach in this agent. The agent follows what’s known as the ReAct pattern (Reason-Act-Observe)

Reason about his documents and requests
Act by using appropriate tools
Observe the results
Repeat as necessary until I’ve fully addressed his needs
"""

# Build Workflow
builder = StateGraph(MessagesState)

# Add Nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))


# Add edges to connect nodes
builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge("tools", "assistant")

# Compile the agent
react_agent = builder.compile()

# Show the agent
graph = react_agent.get_graph(xray=True)
png_bytes = graph.draw_mermaid_png()

# Save image to a file
with open("agent_graph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as agent_graph.png")

os.system("open agent_graph.png")  # macOS


if __name__ == "__main__":
    user_prompt = "Please pre-process the provided image with thresholding to improve readability, and transcribe the recipe in the provided image. Convert to temperature from Fahrenheit to Celsius and all measurement from imperial to metrics."

    messages = [HumanMessage(content=user_prompt)]
    messages = react_agent.invoke({
        "messages": messages,
        "input_file": "/Users/shubhamrathod/PycharmProjects/OCR_AI_Agent/images/chocolate_cake_recipe.png"
        })
    
    
    for m in messages["messages"]:
        m.pretty_print()
