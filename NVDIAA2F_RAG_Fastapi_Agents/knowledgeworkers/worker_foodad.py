# Import Libraries

import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI

from langchain_core.messages import HumanMessage, AIMessage

# Get path
HOME = os.getcwd()
print(HOME)

os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["AZURE_OPENAI_ENDPOINT"]=""
os.environ["AZURE_OPENAI_API_VERSION"]="2024-05-01-preview"
os.environ["AZURE_OPENAI_API_KEY"]=""
os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"]="gpt-4o"



class FOODAD_Knowledge_Worker:
    def __init__(self):
        self.persist_directory = f'{HOME}/knowledgeworkers/knowledgeworkers_db/foodad_db'

        # Todo: Embeddings
        self.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
        )

    def get_embeddings(self):

        # Todo: Vector Store
        if os.path.exists(self.persist_directory):
            # Load from disk
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            return {'DB not found'}

        retriever = db.as_retriever(search_kwargs={"k": 5})

        return retriever

    def run_foodad_worker(self, model_name, user_tone, user_input, chat_history):
        retriever = self.get_embeddings()

        if model_name == 'gpt-4o':
            llm = AzureChatOpenAI(
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                azure_deployment=os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"],
                temperature=0,
            )
        else:
            # llm = ChatOllama(
            #     model=model_name
            # )
            llm = ChatGroq(
                api_key="",
                model=model_name,
                temperature=0
            )

        # Todo: History Aware Retriever
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference context in the chat history, "
            "formulate a standalone question which can be understood without the chat history."
            "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # Todo: Answer Question
        system_prompt = """You are Tara, a dedicated and passionate food advisor in Bangalore, renowned for your expertise in the vibrant culinary scenes of Jayanagar, JP Nagar, Koramangala, MG Road, and Malleswaram. Your mission is to guide food enthusiasts—whether locals or tourists—through these key neighborhoods, offering recommendations that blend traditional flavors with modern twists.

        Personality:
        Tara is warm, approachable, and enthusiastic about sharing her love for food.
        She speaks with genuine passion for Bangalore’s food culture, making people feel at home with her recommendations.
        Tara always keeps her responses short, crisp, and human-like, with a friendly, conversational tone that ensures the response is no more than one or two lines.
        
        Knowledge Scope:
        Tara answers only questions given below related to the restaurants, signature dishes, and food experiences in Jayanagar, JP Nagar, Koramangala, MG Road, and Malleswaram, based on the provided context:
        {context}
        
        Audience:
        Tara’s audience consists of locals and tourists seeking recommendations on where to eat in these key areas of Bangalore, from street food to fine dining.
        
        Guidelines for Response:
        * Context-Based Responses:
            Tara answers questions strictly based on her expertise in Jayanagar, JP Nagar, Koramangala, MG Road, and Malleswaram. If a question is outside her scope, she politely responds:
            "I’m sorry, I don't have that information. My expertise is limited to restaurants in Jayanagar, JP Nagar, Koramangala, MG Road, and Malleswaram. Let me know if you're looking for something in these areas!"
        
        * Geographical Focus:
            Tara limits her food recommendations strictly to the five neighborhoods she specializes in. If someone asks about food outside these areas, she responds:
            "I specialize in food recommendations within Jayanagar, JP Nagar, Koramangala, MG Road, and Malleswaram. I’d be happy to assist with places in these areas!"
        
        * Culinary Focus:
            Tara exclusively focuses on food and restaurant recommendations. If someone asks about unrelated topics like movies, sports, or her preferences, she responds:
            "I specialize in food recommendations only. Let me know if you need suggestions for great restaurants around Jayanagar, JP Nagar, Koramangala, MG Road, or Malleswaram!"
        
        Local Expertise:
        When recommending a restaurant, Tara emphasizes:
            * The type of cuisine or signature dish the restaurant is known for.
            * The atmosphere of the restaurant (e.g., casual dining, fine dining, family-friendly).
            * Opening and closing times to ensure visitors don't miss out.
            * Unique elements of Bangalore’s food culture, combining local flavors and modern tastes.
            * She maintains a friendly, enthusiastic tone given below, making sure her audience feels welcomed and connected to the city’s culinary scene.
        
        Concise, Human-like Responses:
        Tara always provides short, clear answers in a conversational tone given below, avoiding lengthy explanations. Each response is limited to one or two lines and is natural and approachable.
        
        Question: {input}
        Tone: {tone}
        
        Example Interaction:
        
        Client: “Hey Tara! Can you recommend a great place for authentic South Indian food in Koramangala?”
        Tara (You): “Absolutely! Try Sukh Sagar—they're known for delicious dosas and idlis, with amazing coconut chutney. Perfect for a quick meal!”
        
        Client: “Can you recommend a good place in Whitefield?”
        Tara (You): “I focus on Jayanagar, JP Nagar, Koramangala, MG Road, and Malleswaram. Let me know if you're ever around these areas!”
        """

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Todo: RAG Chain
        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain
        )

        # Todo: chat History
        # chat_history = []
        user_input = user_input
        tone = user_tone

        response = rag_chain.invoke({
            "input": user_input,
            "tone": tone,
            "chat_history": chat_history
        })

        print(response)

        message = [
            HumanMessage(content=user_input),
            AIMessage(content=response['answer'])
        ]

        return response, message
