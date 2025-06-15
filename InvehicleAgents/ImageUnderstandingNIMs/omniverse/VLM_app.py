import cv2
import streamlit as st
import numpy as np

import os

import requests, base64

col1, col2 = st.columns(2)

def image_Display(image_file):
    # Read image from the uploaded file
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Create a copy of the original image for the left preview
    img_preview = img.copy()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    col1.image(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB), channels="RGB", caption="Camera's Latest Click", use_column_width=True)




st.title("Visual Language Model")

def get_latest_image(folder_path):
    try:
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
        if not image_files:
            st.warning(f"No images found in the folder: {folder_path}")
            return None

        # Get the latest file by modified time
        latest_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        return os.path.join(folder_path, latest_file)
    except Exception as e:
        st.error(f"Error retrieving the latest image: {e}")
        return None
    


# Function to browse image files from selected camera folder
def browse_images_from_camera(camera_label):
    folder_paths = {
        "Camera 1": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 1",
        "Camera 2": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 2",
        "Camera 3": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 3",
        "Camera 4": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 4",
        "Camera 5": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_01",
        "Camera 6": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_03",
        "Camera 7": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_07",
        "Camera 8": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_16",
        "Camera 9": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_17",
        "Camera 10": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_19",
        "Camera 11": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_20"
    }
    
    selected_folder = folder_paths.get(camera_label)
    
    if selected_folder and os.path.exists(selected_folder):
        # st.write(f"Selected camera folder: {selected_folder}")
        # image_files = [f for f in os.listdir(selected_folder) if f.endswith(('jpg', 'jpeg', 'png'))]
        # selected_image = st.selectbox("Select an image from the folder", image_files)
        selected_image = get_latest_image(selected_folder)
        if selected_image:
            image_path = os.path.join(selected_folder, selected_image)
            with open(image_path, "rb") as img_file:
                image_Display(img_file)
    else:
        st.warning(f"The folder for {camera_label} does not exist or is inaccessible.")

def Run_VLM(Query,camera_label) : 

    invoke_url = "https://ai.api.nvidia.com/v1/vlm/nvidia/vila"
    stream = False

    folder_paths = {
        "Camera 1": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 1",
        "Camera 2": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 2",
        "Camera 3": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 3",
        "Camera 4": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Camera 4",
        "Camera 5": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_01",
        "Camera 6": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_03",
        "Camera 7": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_07",
        "Camera 8": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_16",
        "Camera 9": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_17",
        "Camera 10": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_19",
        "Camera 11": "C:\\Users\\ov-developer\\Documents\\VLM\\Cummins\\pose_estimation_streamlit\\pose_views\\Cummins_Camera_20"
    }

    selected_folder = folder_paths.get(camera_label)
        
    if selected_folder and os.path.exists(selected_folder):
        selected_image = get_latest_image(selected_folder)
        if selected_image:
            image_path = os.path.join(selected_folder, selected_image)
            with open(image_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
    else:
        st.warning(f"The folder for {camera_label} does not exist or is inaccessible.")

    # assert len(image_b64) < 180_000, \
    #   "To upload larger images, use the assets API (see docs)"
    # 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

    headers = {
    "Authorization": "Bearer nvapi-Continue your api key here",
    "Accept": "text/event-stream" if stream else "application/json"
    }

    if (Query==""):
        Q = 'You are an advanced visual model designed to analyze images from factory environments. Your task is to detect safety hazards in the scene. Hazards may include but are not limited to: exposed machinery, improper safety gear, fire risks, spills, electrical hazards, or any situation that could pose danger to workers. Provide a small clear description of the hazards identified.'
    else:
        Q = 'You are an advanced visual model designed to analyze images from factory environments. Your task is to detect safety hazards in the scene. Hazards may include but are not limited to: exposed machinery, improper safety gear, fire risks, spills, electrical hazards, or any situation that could pose danger to workers. Provide a small and clear description of the hazards identified.' + Query

    payload = {
    "messages": [
        {
        "role": "user",
        "content": f'{Q} <img src="data:image/png;base64,{image_b64}" />'
        }
    ],
    "max_tokens": 1024,
    "temperature": 0.20,
    "top_p": 0.70,
    "seed": 50,
    "stream": False,
    }

    response = requests.post(invoke_url, headers=headers, json=payload)

    if stream:
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    else:
            st.write(response.json()["choices"][0]["message"]["content"])
    


# Dropdown to select camera
camera_label = st.selectbox("Select Camera", ["Camera 1","Camera 2", "Camera 3", "Camera 4","Camera 5","Camera 6","Camera 7","Camera 8","Camera 9","Camera 10","Camera 11"])

# Call the function to browse images based on the selected camera
browse_images_from_camera(camera_label)

VLM_Query = st.text_input('Ask Your Query',value="Describe the scene?")
st.button(label="Query!!!", on_click=Run_VLM(VLM_Query,camera_label))






