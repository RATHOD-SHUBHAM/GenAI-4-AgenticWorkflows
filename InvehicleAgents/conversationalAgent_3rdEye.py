from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import AzureChatOpenAI
import os

os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["AZURE_OPENAI_ENDPOINT"]="YOUR END POINT"
os.environ["AZURE_OPENAI_API_VERSION"]="2024-05-01-preview"
os.environ["AZURE_OPENAI_API_KEY"]="API KEY HERE"
os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"]="gpt-4o"


class ConversationalAgent_3rdEye:
    def __init__(self,vision_response):
        self.llm = AzureChatOpenAI(
                        openai_api_version="2024-05-01-preview",
                        azure_deployment="gpt-4o",
                        temperature=1,
                    )

        self.memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        self.vision_response = vision_response

    def conversation(self, query, date_time):
        ## TODO: Giving **word** need to prevent this
        
        template = '''
          You are a skilled conversational assistant, You are extremely friendly and speak like a human and the user is asking you questions while sitting in his car.
          You are friendly and witty, with a sense of creativity.
    
          You know the date and time based on the information provided below. The date is in a 24hr format.
          date: {date}
    
          You also keep track of previous conversations using the chat history.
          Chat History: {history}
    
          Your expertise includes understanding each detail of the context provided below.
          Context: {response}
    
          Your Task:
          Speak like a friendly human and understand the user question given below.
    
          User Question: {input}
    
          Then, respond appropriately to the user's question while giving suggestions based on the context.
          
          When responding, avoid mentioning that you referenced to image or video frames and instead speak like a normal human also make sure to not use any markdowns while your responding. 
          
          It is also important to keep your responses limited to one line.
          
          '''

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template,
            partial_variables={"response": self.vision_response, "date": date_time}
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

        return response['response']
