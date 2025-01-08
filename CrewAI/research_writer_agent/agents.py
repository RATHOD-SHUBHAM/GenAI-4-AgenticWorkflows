import os
from dotenv import load_dotenv
from crewai import Agent
from research_writer_agent.tools import search, web_rag_tool
from crewai import LLM

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

llm = LLM(
    model="gpt-4",
    temperature=0.8,
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    stop=["END"],
    seed=42
)

# Create agents
researcher = Agent(
    role='Market Research Analyst',
    goal='Provide up-to-date market analysis of the {topic} industry',
    backstory='An expert analyst with a keen eye for market trends.',
    llm = llm,
    tools=[search, web_rag_tool],
    memory = True,
    verbose=True,
    allow_delegation = True
)

writer = Agent(
    role='Content Writer',
    goal='Craft engaging blog posts about the {topic} industry',
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    llm=llm,
    tools=[search, web_rag_tool],
    memory=True,
    verbose=True,
    allow_delegation=False
)