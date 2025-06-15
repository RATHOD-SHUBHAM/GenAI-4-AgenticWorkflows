from langchain_ollama import ChatOllama
from langchain.prompts import load_prompt

class LocalLLM:
    def __init__(self, model_name):
        self.model = model_name

    def run_llm(self, user_input):
        # Load the prompt
        prompt = load_prompt('prompts/test_prompt.json')

        # format the prompt to add variable values
        prompt_formatted_str: str = prompt.format(question=user_input)

        llm = ChatOllama(
            model = self.model,
            temperature = 0
        )

        chain = prompt | llm

        response = chain.invoke(prompt_formatted_str)

        # print(response)

        print(response.content)

        return response.content


# if __name__ == '__main__':
#     obj = LocalLLM(model_name = 'mistral:latest')
#     obj.create_knowledge_store(user_input='cat')