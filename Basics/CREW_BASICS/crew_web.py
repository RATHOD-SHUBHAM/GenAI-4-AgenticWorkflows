from crewai_tools import ScrapeWebsiteTool, FileWriterTool, TXTSearchTool
from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize the tool, potentially passing the session
tool = ScrapeWebsiteTool(website_url='https://en.wikipedia.org/wiki/Artificial_intelligence')  

# Extract the text
text = tool.run()
print(text)

# Initialize the tool
file_writer_tool = FileWriterTool()
text = text.encode("ascii", "ignore").decode()
# Write content to a file in a specified directory
result = file_writer_tool._run(filename='ai.txt', content = text, overwrite="True")
print(result)

# Initialize the tool with a specific text file, so the agent can search within the given text file's content
tool = TXTSearchTool(txt='ai.txt')

context = tool.run('What is natural language processing?')

data_analyst = Agent(
    role='Educator',
    goal=f'Based on the context provided, answer the question - What is Natural Language Processing? Context - {context}',
    backstory='You are a data expert',
    verbose=True,
    allow_delegation=False,
    tools=[tool]
)

test_task = Task(
    description="Understand the topic and give the correct response",
    tools=[tool],
    agent=data_analyst,
    expected_output='Give a correct response'
)

crew = Crew(
    agents=[data_analyst],
    tasks=[test_task]
)

output = crew.kickoff()
print(output)