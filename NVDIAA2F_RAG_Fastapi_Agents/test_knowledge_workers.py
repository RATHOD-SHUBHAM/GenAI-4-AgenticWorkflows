from avatar.create_knowledge_store.hr_knoweledge import HR_Knowledge_Worker

hr_obj = HR_Knowledge_Worker()


def hr_test_file(user_input, chat_history):
    response, message = hr_obj.run_hr_worker(model_name = "llama3.2:1b", user_input=user_input, chat_history= chat_history)
    chat_history.extend(message)
    print(response['answer'])
    print(chat_history)



if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("Ask: ")

        if user_input == 'q':
            chat_history = []
            break

        hr_test_file(user_input, chat_history)
