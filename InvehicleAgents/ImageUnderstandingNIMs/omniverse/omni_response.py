from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain_nvidia_ai_endpoints import ChatNVIDIA


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class ConversationalAgent:
    def __init__(self):
        self.llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct",
                              api_key="",
                              temperature=0.2,
                              top_p=0.7,
                              max_tokens=1024
                              )



    def conversation(self, context, query):
        # Todo: Prompt Template
        template = '''System Instruction:
            You are a safety monitoring agent in a factory environment. 
            Your task is to respond to the user’s question regarding the identified hazard.
            
            Incident:
            Incident details: {context}
            
            Your Task:
                * Describe the two major hazards present.
                
                Hazards may include but are not limited to: Exposed machinery parts, Fire risks (e.g., flammable materials near heat sources), Blocked or cluttered walkways, Unattended pallets or objects obstructing pathways, Spills (liquid or other substances), Electrical hazards (exposed wires, overloaded circuits, etc.)  that could pose danger to workers.
                
            You Must
                * Make sure not to respond by stating "as per the provided image."
                * Respond to the user by identifying and describing the primary hazard detected.
                * Keep the response clear and concise.
                * Limit the response to 2 or 3 bullet points, ensuring each point is descriptive and to the point.
                * Use general language, such as "The identified hazards are..."
            
            User Question:
            {input}
          
          '''

        prompt = PromptTemplate(
            input_variables=["input"],
            template=template,
            partial_variables={"context": context}
        )

        # Todo: Create a conversational chain
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=False
        )

        # Todo: Run user query.
        query = query
        response = chain.invoke(query)
        # print(response)

        return response['text']
