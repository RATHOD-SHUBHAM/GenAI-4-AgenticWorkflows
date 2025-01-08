import os
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import Field

import pandas as pd
import numpy as np

HOME = os.getcwd()
# print(HOME)
data_folder = os.path.abspath(os.path.join(HOME, '..', '..', 'data'))


# print(data_folder)

class FeatureEngineering:
    # Load the CSV file
    def analyze_harsh_acceleration(self, file_path, threshold):
        try:
            # Read the CSV file
            data = pd.read_csv(file_path)

            # Check if required columns exist
            if 'engine_speed' not in data.columns or 'timestamp' not in data.columns:
                print("Error: Required columns ('engine_speed', 'timestamp') not found in the CSV file.")
                return

            # Filter rows where engine_speed exceeds 2000
            harsh_acceleration = data[data['engine_speed'] > threshold]

            # Store results in a list
            results = []

            if not harsh_acceleration.empty:
                # print(f"Harsh acceleration detected in {len(harsh_acceleration)} rows.")
                # print("Details of harsh acceleration:")
                for index, row in harsh_acceleration.iterrows():
                    result = {
                        'timestamp': row['timestamp'],
                        'engine_speed': row['engine_speed']
                    }
                    results.append(result)
                    # print(f"Row {index + 1}: Timestamp = {row['timestamp']}, Engine Speed = {row['engine_speed']}")
            else:
                print("No harsh acceleration detected.")

            return results

        except Exception as e:
            print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     fe_obj = FeatureEngineering()
#     file_path = f"{data_folder}/Simulator-2024-12-13 12_31_38.045.csv"
#     results = fe_obj.analyze_harsh_acceleration(file_path=file_path, threshold=2000)
#
#     print(results)
#
#     # Optionally print the results list
#     if results:
#         print("\nSummary of harsh acceleration events:")
#         for event in results:
#             print(event)
