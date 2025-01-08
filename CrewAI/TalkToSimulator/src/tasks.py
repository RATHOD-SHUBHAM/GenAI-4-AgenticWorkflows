from textwrap import dedent
from crewai import Task
from TalkToSimulator.src.tools import AnalyzeHarshSpeedingTool

output_file_path = 'output/output.md'

class AnalyseHarshSpeedingTask:
    def harsh_acceleration_analysis_task(self, agent, dataset):
        return Task(
            description=dedent(f"""
            Your task is to analyze the provided dataset to identify instances of harsh acceleration.

            Dataset: {dataset}

            """),
            expected_output=dedent(f"""
            A one line response if harsh acceleration happened or not.
            """),
            name="harsh_acceleration_analysis_task",
            agent=agent,
            tools=[AnalyzeHarshSpeedingTool().run_analysis],
            async_execution=False
        )

    def harsh_acceleration_response_task(self, agent, acceleration_data):
        return Task(
            description=dedent(f"""
            Your task is to review the provided Harsh Acceleration Data below.
            
            If there is no Harsh Acceleration Data, then simple return stating everything looks perfect.
            
            If Harsh Acceleration Data is present then explain their potential impact on the vehicle.
            The acceleration_data is  a json file with timestamp and engine speed. Use only this information to highlight how this behavior can increase engine wear, reduce fuel efficiency, and pose safety risks. 
            Additionally, provide actionable recommendations to mitigate such occurrences and promote better driving habits.
            Also use the most recent date from the data provided below for creating a report.

            Harsh Acceleration Data: {acceleration_data}
            
            """),
            expected_output=dedent(f"""
            A comprehensive response detailing the effects of harsh acceleration on vehicle performance and safety. Include actionable
            recommendations for mitigating these behaviors and improving driving practices.
            """),
            name="harsh_acceleration_response_task",
            agent=agent,
            verbose= True,
            tools=[AnalyzeHarshSpeedingTool().json_to_text],
            async_execution=False,
            output_file=output_file_path
        )
