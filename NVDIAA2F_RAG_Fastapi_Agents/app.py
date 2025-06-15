# No module named 'streamlit.cli': conda update streamlit


import streamlit as st
from st_audiorec import st_audiorec
import requests
import os
from utils.asr import *

HOME = os.getcwd()

def record_voice():
    # Record Audio
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        # Save the raw audio data to a .wav file
        wav_file_name = f'{HOME}/utils/microphone-results.wav'

        with open(wav_file_name, 'wb') as f:
            f.write(wav_audio_data)  # Write raw bytes directly to the file

        st.write("Audio saved as 'recorded_audio.wav'")


def main():

    # Define the backend API URL
    url = "http://127.0.0.1:8000/generate"

    st.title("Generate Response from FastAPI Backend")


    chat_histories = {
        "hr_worker": [],
        "food_advisor": [],
        "tour_guide": [],
        "wirebond_expert" : []
    }

    # User inputs to populate the query parameters
    model_name = st.selectbox("Model Name", ["gpt-4o", "mistral", "llama3.1", "llama3.2_small", "gemma2"])
    worker_id = st.selectbox("Worker ID", ["hr_worker", "food_advisor", "tour_guide", "wirebond_expert"])
    user_tone = st.selectbox("User Tone", ["Formal", "Funny", "Neutral", "Professional"])


    input_type = st.radio(
        "What's your choice",
        ["***Chat***", "***Voice***"],
        captions=[
            "Type user question.",
            "Talk to me.",
        ],
    )

    if input_type == "***Chat***":
        user_input = st.text_area("User Input", value="Can i wear shorts today?")

        # Button to trigger API call
        if st.button("Generate Response"):
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
            response = requests.post(url, headers=headers, params=params)

            # Check if the response is successful and display the result
            if response.status_code == 200:
                st.success("Response received:")
                st.json(response.json())  # Display the JSON response
            else:
                st.error(f"Error: {response.status_code}\n{response.text}")

    elif input_type == "***Voice***":

        # Record Audio
        audio_file_path = record_voice()
        print(audio_file_path)

        user_input = run_asr()
        st.write(user_input)

        # Button to trigger API call
        if st.button("Generate Response"):
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
            response = requests.post(url, headers=headers, params=params)

            # Check if the response is successful and display the result
            if response.status_code == 200:
                st.success("Response received:")
                st.json(response.json())  # Display the JSON response
            else:
                st.error(f"Error: {response.status_code}\n{response.text}")


if __name__ == '__main__':
    main()