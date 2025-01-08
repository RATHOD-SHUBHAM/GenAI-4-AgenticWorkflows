from textwrap import dedent
from crewai import Task
from EmailMeetingPreparation.src.tools import search_tool,send_email

summary_output_file_path = 'summary_briefing.txt'
meeting_output_file_path = 'meeting_briefing.txt'
email_output_file = 'email_draft.txt'


class EmailMeetingPrepTask:
    # Todo: Add Tasks Here

    def research_task(self, agent, meeting_participants, meeting_context):
        return Task(
            description=dedent(f"""
                Conduct comprehensive research on each of the individuals and companies involved in the upcoming meeting. 
                Gather information on recent news, achievements, professional background, their linkedin Information and any relevant business activities.

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
            async_execution=False,
            output_file=meeting_output_file_path
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
            output_file=summary_output_file_path
        )

    def email_composition_task(self, agent,  meeting_context, meeting_objective):
        return Task(
            description=dedent(f"""
                Write an email that greets the recipient in a professional and coherent manner. The email should have a clear and polite tone, starting with a friendly greeting and following with the main content as shared.
    
                Please ensure the structure includes:
                - A greeting that addresses the recipient appropriately (e.g., "Dear Shubham" or "Hello Shubham")
                - Do not include a subject field.
                - A clear introduction that briefly sets the context or purpose of the email
                - The provided content, presented in a well-organized format
                - A closing sentence that either summarizes the key points or invites a response, followed by a professional sign-off (e.g., "Best regards from SmgLab," "Sincerely SmgLab," etc.)
                
                Meeting Context: {meeting_context}
                Meeting Objective: {meeting_objective}
    
            """),
            expected_output=dedent(f"""
                A well-structured email with the following components:
                - A friendly greeting
                - An introductory sentence setting the tone for the email
                - A detailed yet concise presentation of the provided content
                - A polite closing with a call to action or invitation for further discussion
                - A professional sign-off
            """),
            name="email_composition_task",
            agent=agent,
            output_file=email_output_file,
            async_execution=False
        )

    def send_email_task(self, agent):
        return Task(
            description=dedent(f"""
                    Your task is to send an email with the specified body content using the given tool.
                    """),
            expected_output=dedent(f"""
                    A confirmation that the email was successfully sent.
                    """),
            name="send_email_task",
            agent=agent,
            tools=[send_email],
            async_execution=False
        )

