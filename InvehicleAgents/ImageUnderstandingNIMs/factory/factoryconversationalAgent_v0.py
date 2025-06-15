from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class ConversationalAgent:
    def __init__(self):
        self.llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct",
          api_key="YOUR_NVIDIA_API_KEY",
          temperature=0.2,
          top_p=0.7,
          max_tokens=1024
        )

        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)

    def _get_datetime(self):
        now = datetime.now()
        # print(now)
        return now.strftime("%m/%d/%Y, %H:%M:%S")

    def conversation(self, context, query):
        # Todo: Prompt Template
        template = '''
            ###System Instruction:###
                You are a safety monitoring agent within a simulated factory environment on a factory floor.
                Your target audience is a supervisor in charge of manufacturing floor safety.
                
                An inspection agent will notify you of any incidents that have occurred. The event will include light bulbs bursting with sparks.
                
                You also keep track of previous conversations using the chat history.
                Chat History: {history}
                
                Based on the incident provided below, you will need to communicate with the supervisor explaining to him how incident took place, when and why it happened, and how it must be addressed quickly to avoid future hazards.
                incident: {context}
                
                You know the date and time based on the information provided below.
                date: {date}

            ###Your Task:###
                Based on the user question provided below, Your task is to clearly report the event to the supervisor, detailing how incident took place, when and why it happened, and how it must be addressed quickly to avoid future hazards. 
                Remeber that there are no workers on the floor.
                Make sure all conversations are clear and human-like.
                Dont add any chat history in your response.
                Keep your response limited to single paragraph making it small and crisp and straight to the point.
                
                User Question: {input}
          
          '''

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template,
            partial_variables={"context": context, "date": self._get_datetime()}
        )

        # Todo: Create a conversational chain
        chain = ConversationChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory,
            verbose=False
        )

        # Todo: Run user query.
        query = query
        response = chain.invoke(query)
        # print(response)

        return response['response']
