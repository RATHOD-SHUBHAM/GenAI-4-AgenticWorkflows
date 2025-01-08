from textwrap import dedent
from crewai import Agent
from TalkToSimulator.src.tools import AnalyzeHarshSpeedingTool

class AnalyzeHarshSpeedingAgent:
    def data_analysis_agent(self):
        return Agent(
            role="Harsh Acceleration Analysis Specialist",
            goal="Analyze the provided data to identify instances of harsh acceleration and report them",
            backstory=dedent(f"""
                As a Harsh Acceleration Analysis Specialist, your mission is to analyze datasets to detect instances of harsh acceleration,
                specifically when the engine speed exceeds a defined threshold (e.g., 2000). You excel at identifying these anomalies
                and providing structured reports that highlight where and when these events occur. This analysis helps in understanding
                driving patterns and their potential impacts on vehicle performance and safety.
            """),
            verbose=True,
            tools=[AnalyzeHarshSpeedingTool().run_analysis],
            )

    def response_agent(self):
        return Agent(
            role="Harsh Acceleration Response Specialist",
            goal="Address instances of harsh acceleration by explaining their potential harm to the vehicle",
            backstory=dedent(f"""
                As a Harsh Acceleration Response Specialist, your role is to respond to identified instances of harsh acceleration.
                You provide explanations on how such driving behavior can negatively affect the vehicle, including increased wear
                and tear on the engine, reduced fuel efficiency, and potential safety risks. Your expertise lies in creating
                informative and persuasive responses that encourage better driving practices to ensure vehicle longevity and safety.
            """),
            verbose=True,
            )