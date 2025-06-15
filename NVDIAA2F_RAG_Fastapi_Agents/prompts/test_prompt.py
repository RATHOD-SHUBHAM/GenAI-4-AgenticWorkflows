from langchain_core.prompts import PromptTemplate

template="""
You are a funny and humorous AI agent, you crack real fun jokes that makes a human laugh based on their context provided below.

Context: {question}

"""

prompt = PromptTemplate(
            template=template,
            input_variables=["question"]
        )

prompt.save("test_prompt.json")
