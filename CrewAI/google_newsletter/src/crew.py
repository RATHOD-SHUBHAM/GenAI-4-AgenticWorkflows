from crewai import Crew, Process
from google_newsletter.src.agents import NewsLetterAgents
from google_newsletter.src.tools import SearchTool
from google_newsletter.src.tasks import NewsLetterTasks

import os
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "google_news_letter_agent"


def main():
    print("## Welcome to Google News Letter")
    print('-------------------------------')
    topic = input("What are you interested in today?\n")
    objective = input("What is your objective behind the news?\n")

    # Todo: Initialize task and agents
    tasks = NewsLetterTasks()
    agents = NewsLetterAgents()

    # Todo: Create agents
    news_journalist_agent = agents.news_journalist_agent()
    critique_agent = agents.critique_agent()
    news_letter_writer_agent = agents.news_letter_writer_agent()

    # Todo: Create tasks
    news_collection_task = tasks.news_collection_task(agent=news_journalist_agent, topic=topic, objective=objective)
    news_critique_task = tasks.news_critique_task(agent=critique_agent, topic=topic, objective=objective)
    newsletter_writing_task = tasks.newsletter_writing_task(agent=news_letter_writer_agent, objective=objective)


    # Todo: Pass information from the other  task
    """This will be appended to Task Params/Attribute for the task."""
    newsletter_writing_task.context = [news_collection_task, news_critique_task]

    # Todo: Assemble the crew
    crew = Crew(
        agents=[news_journalist_agent, critique_agent, news_letter_writer_agent],
        tasks=[news_collection_task, news_critique_task, newsletter_writing_task],
        process=Process.sequential,
        memory=True,
        verbose=True
    )

    # Todo: Run Crew
    result = crew.kickoff()
    print('\n\n')
    print('-------------------------------')
    print("result: \n")
    print(result)


if __name__ == '__main__':
    load_dotenv()
    main()