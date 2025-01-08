from crewai.tools import tool
from TalkToSimulator.src.utils.feature_engineering import FeatureEngineering
import os
import json

HOME = os.getcwd()
# print(HOME)
ROOT = os.path.dirname(HOME)
print(ROOT)


class AnalyzeHarshSpeedingTool:

    @tool("run_analysis")
    def run_analysis():
        """Analyze the CVS file to detect if there was harsh speeding."""
        dataset = "/Users/shubhamrathod/PycharmProjects/crewAI/TalkToSimulator/data/Simulator-2024-12-13 12_31_38.045.csv"
        results = FeatureEngineering().analyze_harsh_acceleration(file_path=dataset, threshold=2000)
        print(results)

        # todo: save in folder
        # Specify the file path where you want to save the JSON data
        output_file_path = '/Users/shubhamrathod/PycharmProjects/crewAI/TalkToSimulator/src/output/data.json'
        # Open the file in write mode and write the data
        with open(output_file_path, 'w') as json_file:
            # Write the list to the JSON file
            json.dump(results, json_file, indent=4)

        if results is not None:
            return "harsh acceleration identified"
        else:
            return "Normal"

    @tool("json_to_text")
    def json_to_text():
        """Converts a JSON file to a string."""
        json_file = '/Users/shubhamrathod/PycharmProjects/crewAI/TalkToSimulator/src/output/data.json'

        with open(json_file, 'r') as f:
            data = json.load(f)

        return json.dumps(data, indent=4)


