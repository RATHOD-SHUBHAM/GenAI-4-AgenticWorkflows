from crewai import Crew, Process
from research_writer_agent.agents import researcher, writer
from research_writer_agent.task import researcher_task, writer_task


# Crew Formation
crew = Crew(
    agents=[researcher, writer],
    tasks=[researcher_task, writer_task],
    process = Process.sequential,
    memory=True,
    cache=True,
    share_crew=True,
    verbose=True
)

# Run Crew
inputs = {'topic': 'AI in healthcare'}
result = crew.kickoff(inputs=inputs)
print(result)