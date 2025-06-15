from avatar.create_knowledge_store.hr_knoweledge import HR_Knowledge_Worker

# Object for each worker
hr_obj = HR_Knowledge_Worker()

chat_histories = {
    "hr_worker": [],
    "food_advisor": [],
    "worker3": []
}

current_worker = None


def get_chat_history(worker_id):
    global current_worker

    if worker_id != current_worker:
        # Clear chat history if switching to a new worker
        chat_histories[worker_id] = []
        # Switch to a new worker
        current_worker = worker_id

    return chat_histories[worker_id]


def run_llm_worker(model_name, worker_id, user_input):
    if worker_id == "hr_worker":
        chat_history = get_chat_history(worker_id)
        # print(chat_history)
        response, message = hr_obj.run_hr_worker(model_name=model_name, user_input=user_input,
                                                 chat_history=chat_history)

        chat_history.extend(message)

        return response['answer']

    elif worker_id == "food_advisor":
        chat_history = get_chat_history(worker_id)
        return "cleared history"


# IF HR_RUN_HR
# IF Tour Guide -> Run Tour guide
# IF Food Expert -> Run food expert
