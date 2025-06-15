import streamlit as st
from hazard_analyst import ImageAnalyst
from omni_response import ConversationalAgent
# from omniverse.hazard_analyst import ImageAnalyst
# from omniverse.omni_response import ConversationalAgent
import os

HOME = os.getcwd()
# print(HOME)

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1/bin/ffmpeg"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def startConversation(user_input, summary):
    conversation_obj = ConversationalAgent()

    while True:
        # user_query = input("Ask: ")

        user_query = user_input

        if user_query == 'q' or user_query == 'quit' or user_query == 'end':
            print("Stopping")
            break
        elif user_query == " ":
            print('Ask something !!!!!')
            continue
        else:
            response = conversation_obj.conversation(context=summary, query=user_query)
            # print(response)
            return response


def summarize(file_path):
    # Create an object for image analyst
    img_obj = ImageAnalyst()

    # Pass the image to the pipeline function
    image_analysis_result = img_obj.startAnalysis(file_path)

    # print(image_analysis_result)

    return image_analysis_result


def main():
    # Todo: Step 1: Request user for a input
    file_path = f'{HOME}/oil_spill.png'

    if not os.path.exists(file_path):
        print("The specified file does not exist.")
        return


    # Todo: Preprocess the file
    summary = summarize(file_path)

    # Todo: Start Conversation
    user_input = st.text_input("Ask: ", "What is wrong in the scene")
    if st.button("Run"):
        response = startConversation(user_input = user_input, summary=summary)

        # Streamlit
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input Image")
            st.image(file_path)

        with col2:
            st.subheader("Description")
            st.write(response)


if __name__ == '__main__':
    main()
