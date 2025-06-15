from factory_ImageAnalyst_ver_0 import ImageAnalyst
from factoryconversationalAgent_v0 import ConversationalAgent
import os

HOME = os.getcwd()

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/5.1/bin/ffmpeg"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def startConversation(summary):
    conversation_obj = ConversationalAgent()

    print("Alert: An incident occurred")
    while True:
        user_query = input("Investigate: ")

        if user_query == 'q' or user_query == 'quit' or user_query == 'end':
            print("Stopping")
            break
        elif user_query == " ":
            print('Ask something !!!!!')
            continue
        else:
            response = conversation_obj.conversation(context=summary, query=user_query)
            print(response)


def summarize(file_path):
    # Create an object for image analyst
    img_obj = ImageAnalyst()

    # Pass the image to the pipeline function
    image_analysis_result = img_obj.startAnalysis(file_path)

    # print(image_analysis_result)

    return image_analysis_result


def main():
    # Todo: Step 1: Request user for a input
    file_path = f'{HOME}/test_1.jpeg'

    if not os.path.exists(file_path):
        print("The specified file does not exist.")
        return


    # Todo: Preprocess the file
    summary = summarize(file_path)

    # Todo: Start Conversation
    startConversation(summary=summary)


if __name__ == '__main__':
    main()
