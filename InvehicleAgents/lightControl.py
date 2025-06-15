# Todo: Import Libraries
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
# from langchain.chat_models import AzureChatOpenAI
from langchain_community.chat_models import AzureChatOpenAI
import os
os.environ["OPENAI_API_TYPE"] = "azure_ad"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_AZURE_OPENAI_ENDPOINT"  # e.g., https://your-resource-name.openai.azure.com/
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-05-01-preview"
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_AZURE_OPENAI_API_KEY"  # Your Azure OpenAI API key
os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"] = "gpt-4o"

# Todo: Call LLM
llm = AzureChatOpenAI(
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    azure_deployment=os.environ["AZURE_OPENAI_GPT4O_MODEL_NAME"],
    temperature=1,
)

# Todo: Few Shot Prompt
def mood_llm_call(query,country):
    examples = [
        {
            "query": "I had a bad day at work",
            "answer": "Aw man, I hate to hear that you're going through a tough time. But you know what? But hey, listen, chin up! I'm here for you, Let's turn things around and bring some positivity into your day!. : Sad"
        },
        {
            "query": "Nothing seems to working these days",
            "answer": "I understand how you feel. It can be frustrating when things don't seem to go our way. Have you tried taking a break or doing something you enjoy to take your mind off things? Sometimes it can help to take a step back and come back to the problem with a fresh perspective. : Frustrated"
        },
        {
            "query": "Lets head to work",
            "answer": "Alright, have a good day at work! If you need anything else, don't hesitate to ask. : Neutral"
        },
        {
            "query": "You know I went to watch Northern Lights last year at this time. It was beautiful",
            "answer": "It's always great to reminisce about good memories, isn't it? I'm glad you have that wonderful experience of watching the Northern Lights to look back on. Let me help you relive that moment again. : Aurora"
        },
        {
            "query": "I am exited about my upcoming dubai trip and its dessert",
            "answer": "I'm glad to hear that you're excited about your upcoming trip to Dubai and the desert adventure. Dubai has many amazing experiences to offer, and the desert safari is definitely one of them. Let me give you a little boost of excitement. : Desert Storm"
        },
        {
            "query": "Can you surpirse me",
            "answer": "Sure, close your eyes and imagine yourself being suddenly surrounded by colorful fireworks in the night sky, while feeling the excitement and thrill of the moment. : Surprise"
        },

    ]

    example_template = """
        Human: {query}
        AI: {answer}
    """

    example_prompt = PromptTemplate(
        input_variables=["country", "query", "answer"],
        template=example_template
    )

    prefix = """
    You are a talkative, friendly, and compassionate assistant. You are super expressive, You now live in {country}, and all of your assertions are culturally specific.
    Your specialty involves understanding human emotions.

    Your objective is to understand what the human is attempting to express and then map it to one of the mood, celebration or event listed below.
    If it is not a mood or celebration but rather anything associated with an event, assign it to the appropriate event listed below.

    You must also respond to the user in a courteous and empathic manner and restric your response to a single sentence.

    moods : ["Happy", "Sad", "Tensed", "Frustrated", "Bored", "Romantic", "Energetic", "Lazy", "Calm", "Neutral", "Surprise"].
    
    celebration : ["Christmas", "Cherry Blossom" , "Halloween", "Easter", "Oktoberfest", "Weihnachtsmarkt", "Unity Day", "Karneval","Holi", "Diwali", "Independence Day", "Ratha Yatra", "IPL"]

    events : ["Aurora", "Jungle Safari", "Desert Storm", "Beach Drive"]

    Here are some examples:
    """

    suffix = """
        User: {query}
        AI:
    """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator='\n\n'
    )

    chain = LLMChain(llm=llm,
                    prompt=few_shot_prompt_template,
                    verbose=False)

    
    result = chain.run({'country': country,
                        'query': query})

    return result

## Removed from list for German context
# "Holi", "Diwali", "Independence Day", "Ratha Yatra", "IPL"