from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class LocalLLM:
    def __init__(self, model_name):
        self.model = model_name

    def run_llm(self, user_input):
        system_prompt = "You are a funny and humarous AI assistant"
        user_prompt = "Tell me a joke about {user_input}"

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             system_prompt),
            ("user",
             user_prompt)
        ])

        llm = ChatOllama(
            model = self.model,
            temperature = 0
        )

        chain = prompt | llm

        response = chain.invoke({"user_input" : user_input})

        # print(response.content)

        return response.content


if __name__ == '__main__':
    obj = LocalLLM(model_name = 'mistral:latest')
    obj.run_llm(user_input='cat')