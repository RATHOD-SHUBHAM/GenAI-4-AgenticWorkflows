from dotenv import load_dotenv
from crewai import Crew, Process

from MeetingPreparation.src.task import MeetingPrepTask
from MeetingPreparation.src.agents import MeetingPrepAgents

def main():
    print("## Welcome to the Meeting Prep Crew")
    print('-------------------------------')
    meeting_participants = input("What are the emails for the participants (other than you) in the meeting?\n")
    meeting_context = input("What is the context of the meeting?\n")
    meeting_objective = input("What is your objective for this meeting?\n")

    # Todo: Initialize task and agents
    tasks = MeetingPrepTask()
    agents = MeetingPrepAgents()

    # Todo: Create agents
    research_agent = agents.research_agent()
    industry_analysis_agent = agents.industry_analysis_agent()
    meeting_strategy_agent = agents.meeting_strategy_agent()
    summary_and_briefing_agent = agents.summary_and_briefing_agent()

    # Todo: Create tasks
    research_task = tasks.research_task(agent=research_agent, meeting_participants=meeting_participants, meeting_context=meeting_context)
    industry_analysis_task = tasks.industry_analysis_task(agent=industry_analysis_agent, meeting_participants=meeting_participants, meeting_context=meeting_context)
    meeting_strategy_task = tasks.meeting_strategy_task(agent=meeting_strategy_agent, meeting_context=meeting_context, meeting_objective=meeting_objective)
    summary_and_briefing_task = tasks.summary_and_briefing_task(agent=summary_and_briefing_agent, meeting_context=meeting_context, meeting_objective=meeting_objective)

    # Todo: Pass information from the other  task
    """This will be appended to Task Params/Attribute for the task."""
    meeting_strategy_task.context = [research_task, industry_analysis_task]
    summary_and_briefing_task.context = [research_task, industry_analysis_task, meeting_strategy_task]

    # Todo: Assemble the crew
    crew = Crew(
        agents = [research_agent, industry_analysis_agent, meeting_strategy_agent, summary_and_briefing_agent],
        tasks = [research_task, industry_analysis_task, meeting_strategy_task, summary_and_briefing_task],
        process = Process.sequential,
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