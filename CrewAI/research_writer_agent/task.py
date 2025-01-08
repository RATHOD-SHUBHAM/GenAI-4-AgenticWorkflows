import os
from crewai import Task
from research_writer_agent.agents import researcher, writer
from research_writer_agent.tools import search, web_rag_tool

HOME = os.getcwd()
output_file_path = 'output.md'

researcher_task = Task(
    description = """
        Conduct a thorough research about {topic}.
        Make sure you find any interesting and relevant information given
        the current year is 2024.
    """,
    expected_output = """
        A list with 10 bullet points of the most relevant information about {topic}
    """,
    name = "research task",
    agent = researcher,
    tools = [search, web_rag_tool]
)

writer_task = Task(
    description = """
        Review the context you got and expand each topic into a full section for a report.
        Make sure the report is detailed and contains any and all relevant information.
    """,
    expected_output = """
        A fully fledge three paragraph reports with three mains topics, each with a full section of information.
        Formatted as markdown
    """,
    name = "writing task",
    agent = writer,
    tools = [search, web_rag_tool],
    async_execution = False,
    output_file = output_file_path
)