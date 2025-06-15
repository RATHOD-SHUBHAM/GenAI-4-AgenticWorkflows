import gradio as gr
import requests
import os
import time

# Define the backend API URLs
stt_url = "http://127.0.0.1:8000/STT"
generate_url = "http://127.0.0.1:8000/generate"


# Function to handle the STT API request
def perform_speech_to_text(user_voice):
    if user_voice is None:
        return "No audio file recorded.", None

    # Save the recorded voice to a temporary file
    voice_file_path = f"temp_voice_{time.time()}.wav"
    with open(voice_file_path, "wb") as f:
        f.write(user_voice)

    # Open the saved file and send to the STT endpoint
    with open(voice_file_path, "rb") as voice_file:
        files = {'audio_file': voice_file}
        response = requests.post(stt_url, files=files)

    # Check if the STT request was successful
    if response.status_code == 200:
        stt_result = response.json().get("text")
        return stt_result, voice_file_path
    else:
        return f"STT Error: {response.status_code}", None


# Function to handle the generate API request
def generate_response(model_name, worker_id, user_tone, user_input):
    # Prepare the query parameters from user input
    params = {
        "model_name": model_name,
        "worker_id": worker_id,
        "user_tone": user_tone,
        "user_input": user_input
    }

    # Set request headers
    headers = {
        'Content-Type': 'application/json'
    }

    # Make the POST request to the FastAPI backend with parameters
    response = requests.post(generate_url, headers=headers, params=params)

    # Handle and display the response from the API
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code}


# Gradio Interface
with gr.Blocks() as demo:
    # Step 1: Record the voice
    stt_instruction = gr.Textbox(label="Speech-to-Text Instruction",
                                 value="Record your voice to convert speech to text.", interactive=False)
    user_voice = gr.Audio(label="Record your voice", sources="microphone", type="filepath")

    # Step 2: Convert voice to text (STT)
    stt_result = gr.Textbox(label="Speech-to-Text Result",
                            placeholder="Your text will appear here after speech-to-text conversion.")
    convert_btn = gr.Button("Convert Speech to Text")

    # Step 3: Inputs for generating response
    model_name = gr.Textbox(label="Model Name", value="llama3.1")
    worker_id = gr.Textbox(label="Worker ID", value="hr_worker")
    user_tone = gr.Dropdown(label="User Tone", choices=["Formal", "Funny", "Neutral"], value="Funny")
    user_input = gr.Textbox(label="User Input (from STT result)",
                            placeholder="This will be auto-filled from STT result")

    # Step 4: Generate Response Button
    generate_btn = gr.Button("Generate Response")

    # Output display area
    api_response = gr.JSON(label="Generated Response")


    # Link the STT button to the function
    def convert_and_fill(user_voice):
        text, _ = perform_speech_to_text(user_voice)
        return text, text


    convert_btn.click(convert_and_fill, inputs=[user_voice], outputs=[stt_result, user_input])

    # Link the generate button to the response generator function
    generate_btn.click(generate_response,
                       inputs=[model_name, worker_id, user_tone, user_input],
                       outputs=api_response)

# Launch the Gradio app
demo.launch()
