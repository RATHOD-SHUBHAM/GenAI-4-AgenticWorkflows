import os
from openai import OpenAI


# Load the environments
HOME = os.getcwd()
ROOT = os.path.dirname(HOME)
BASE_DIR = os.path.dirname(HOME)
# Todo: My openai key
os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

# Todo: API call
def run_asr():
    client = OpenAI()
    audio_file_path = f'{HOME}/utils/microphone-results.wav'
    audio_file = open(audio_file_path, "rb")

    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    print(transcription)

    return transcription