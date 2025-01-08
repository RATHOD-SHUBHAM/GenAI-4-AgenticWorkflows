'''

They act as the backbone to our flow.

They are kind of todo list that needs to be accomplished.

Tip: Keep your description detailed
'''

from textwrap import dedent
from crewai import Task
from MeetingPreparation.src.tools import search_tool

output_file_path = 'brefing.md'

class MeetingPrepTask:
    # Todo: Add Tasks Here

    def research_task(self, agent, meeting_participants, meeting_context):
        return Task(
            description=dedent(f"""
                Conduct comprehensive research on each of the individuals and companies involved in the upcoming meeting. 
                Gather information on recent news, achievements, professional background, and any relevant business activities.

                Participants: {meeting_participants}
                Meeting Context: {meeting_context}
                """),
            expected_output=dedent(f"""
                    A detailed report summarizing key findings about each participant and company, highlighting information that could be relevant for the meeting.
                    """),
            name="research_task",
            agent=agent,
            tools=[search_tool],
            async_execution=True
        )

    def industry_analysis_task(self, agent, meeting_participants, meeting_context):
        return Task(
            description=dedent(f"""
                        Analyze the current industry trends, challenges, and opportunities relevant to the meeting's context. Consider market reports, recent developments, and expert opinions to provide a comprehensive overview of the industry landscape. 
                        Research relevant statistics, case studies, and market projections that could influence the meeting's objectives.

                        Participants: {meeting_participants}
                        Meeting Context: {meeting_context}
                        """),
            expected_output=dedent(f"""
                        A comprehensive industry report providing insights into major trends, challenges, and opportunities that could shape the upcoming meeting's discussions. 
                        The analysis should be tailored to the participants' sector and their potential impact on the meeting.
                        """),
            name="industry_analysis_task",
            agent=agent,
            tools=[search_tool],
            async_execution=True
        )

    def meeting_strategy_task(self, agent, meeting_context, meeting_objective):
        return Task(
            description=dedent(f"""
                        Develop a strategic plan for the meeting based on the research gathered on participants and the industry analysis. 
                        This plan should identify key discussion points, goals, and potential challenges. It should also include suggested negotiation strategies, possible talking points, and how to approach sensitive topics. 
                        
                        Meeting Context: {meeting_context}
                        Meeting Objective: {meeting_objective}
                        """),
            expected_output=dedent(f"""
                            A well-structured meeting strategy outlining the key objectives, discussion points, potential risks, and negotiation tactics. 
                            The plan will also include recommendations for how to leverage the research and industry insights to drive the meeting towards a successful outcome.
                        """),
            name="meeting_strategy_task",
            agent=agent,
            async_execution=False
        )

    def summary_and_briefing_task(self, agent, meeting_context, meeting_objective):
        return Task(
            description=dedent(f"""
                        Create a concise and actionable summary of the meeting strategy. 
                        This includes a briefing document for all stakeholders, highlighting the meeting objectives, key talking points, and strategic approaches to be used. 
                        Ensure that the summary is clear, easy to follow, and aligned with the meeting's goals.
                        Meeting Context: {meeting_context}
                        Meeting Objective: {meeting_objective}
                        """),
            expected_output=dedent(f"""
                        A succinct briefing document summarizing the meeting strategy. This should include key discussion points, critical objectives, and strategic approaches tailored to the needs of each stakeholder. 
                        The summary should be easily digestible and ready for distribution to relevant parties.
                        """),
            name="summary_and_briefing_task",
            agent=agent,
            async_execution=False,
            output_file=output_file_path
        )

