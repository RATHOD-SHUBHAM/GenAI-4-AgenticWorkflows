from fastapi import FastAPI, HTTPException, File, UploadFile
from typing import Annotated
import uvicorn

# Import class
# from ollama_test_chatprompt import LocalLLM
# from avatar.avatar_main import run_llm_worker
from knowledgeworkers.workers_main import run_llm_worker

app = FastAPI()

# get_model_name = {
#     "gpt-4o": "gpt-4o",
#     "mistral": "mistral:latest",
#     "llama3.1": "llama3.1:latest",
#     "llama3.2_small": "llama3.2:1b"
# }

# Groq Model
get_model_name = {
    "gpt-4o": "gpt-4o",
    "mistral": "mixtral-8x7b-32768",
    "llama3.1": "llama-3.1-8b-instant",
    "llama3.2_small": "llama-3.2-1b-preview",
    "gemma2": "gemma2-9b-it"
}


@app.get('/')
def greeting():
    return {"message": "Hello from Shubham"}


@app.post("/STT")
def speechtotext(audio_file: UploadFile = File(...)):
    try:
        test_audio_file = audio_file

        return {"Response": test_audio_file}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
def generate_response(model_name: str, worker_id: str, user_tone: str, user_input: str):
    try:
        print(model_name, worker_id, user_tone, user_input)
        model_name = get_model_name[model_name]

        response = run_llm_worker(model_name, worker_id, user_tone, user_input)

        return {"Response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main:app", host='0.0.0.0', port=8000, reload=True)
