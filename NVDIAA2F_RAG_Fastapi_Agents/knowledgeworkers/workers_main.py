from knowledgeworkers.worker_hr import HR_Knowledge_Worker
from knowledgeworkers.worker_foodad import FOODAD_Knowledge_Worker
from knowledgeworkers.worker_tourgd import TOURGD_Knowledge_Worker
from knowledgeworkers.worker_wirebond import Wirebond_Knowledge_Worker

# Object for each worker
hr_obj = HR_Knowledge_Worker()
foodad_obj = FOODAD_Knowledge_Worker()
tourgd_obj = TOURGD_Knowledge_Worker()
wirebond_obj = Wirebond_Knowledge_Worker()

# Todo: Worker ID Must Match here
chat_histories = {
    "hr_worker": [],
    "food_advisor": [],
    "tour_guide": [],
    "wirebond_expert" : []
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


def run_llm_worker(model_name, worker_id, user_tone, user_input):
    if worker_id == "hr_worker":
        # Todo: Get the history
        chat_history = get_chat_history(worker_id)
        # print(chat_history)

        # Todo: Run the model
        response, message = hr_obj.run_hr_worker(model_name=model_name, user_tone=user_tone, user_input=user_input,
                                                 chat_history=chat_history)

        chat_history.extend(message)

        return response['answer']

    # Todo: Food Advisor
    elif worker_id == "food_advisor":
        chat_history = get_chat_history(worker_id)
        # print(chat_history)

        response, message = foodad_obj.run_foodad_worker(model_name=model_name, user_tone=user_tone,
                                                         user_input=user_input,
                                                         chat_history=chat_history)

        chat_history.extend(message)

        return response['answer']

    elif worker_id == "tour_guide":
        chat_history = get_chat_history(worker_id)
        # print(chat_history)

        response, message = tourgd_obj.run_tourgd_worker(model_name=model_name, user_tone=user_tone,
                                                         user_input=user_input,
                                                         chat_history=chat_history)

        chat_history.extend(message)

        return response['answer']

        # Todo: Food Advisor
    elif worker_id == "wirebond_expert":
        chat_history = get_chat_history(worker_id)
        # print(chat_history)

        response, message = wirebond_obj.run_foodad_worker(model_name=model_name, user_tone=user_tone,
                                                         user_input=user_input,
                                                         chat_history=chat_history)

        chat_history.extend(message)

        return response['answer']
