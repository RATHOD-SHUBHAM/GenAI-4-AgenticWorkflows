from crewai import Crew, Process
from TalkToSimulator.src.agents import AnalyzeHarshSpeedingAgent
from TalkToSimulator.src.tasks import AnalyseHarshSpeedingTask

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Talk To Simulator"


def main():
    print("## Welcome to live simulator")
    print('-------------------------------')
    dataset = "/Users/shubhamrathod/PycharmProjects/crewAI/TalkToSimulator/data/Simulator-2024-12-13 12_31_38.045.csv"
    acceleration_data = '/Users/shubhamrathod/PycharmProjects/crewAI/TalkToSimulator/src/output/data.json'

    # Todo: Initialize task and agents
    tasks = AnalyseHarshSpeedingTask()
    agents = AnalyzeHarshSpeedingAgent()

    # Todo: Create agents
    data_analysis_agent = agents.data_analysis_agent()
    response_agent = agents.response_agent()

    # Todo: Create tasks
    harsh_acceleration_analysis_task = tasks.harsh_acceleration_analysis_task(agent=data_analysis_agent, dataset = dataset)
    harsh_acceleration_response_task = tasks.harsh_acceleration_response_task(agent=response_agent, acceleration_data=acceleration_data)


    # Todo: Pass information from the other  task
    """This will be appended to Task Params/Attribute for the task."""
    harsh_acceleration_response_task.context = [harsh_acceleration_analysis_task]

    # Todo: Assemble the crew
    crew = Crew(
        agents=[data_analysis_agent, response_agent],
        tasks=[harsh_acceleration_analysis_task, harsh_acceleration_response_task],
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

    print('\n\n')
    print('-------------------------------')

    print(f"""
        Task completed!
        Task: {harsh_acceleration_analysis_task.output.description}
        Output: {harsh_acceleration_analysis_task.output.raw}
    """)

    print(f"""
            Task completed!
            Task: {harsh_acceleration_response_task.output.description}
            Output: {harsh_acceleration_response_task.output.raw}
        """)


if __name__ == '__main__':
    load_dotenv()
    main()