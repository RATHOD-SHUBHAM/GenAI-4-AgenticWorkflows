from dotenv import load_dotenv
from crewai import Crew, Process

from ChatConversation.src.tasks import ChatConversationalTask
from ChatConversation.src.agents import ChatConversationalAgents


def main(user_input, kick_off):
    # print("## Welcome to my Chat Application")
    # print('-------------------------------')
    #
    # user_input = input("Go ahead, Ask your question: ")

    # Todo: Initialize task and agents
    tasks = ChatConversationalTask()
    agents = ChatConversationalAgents()

    # Todo: Create agents
    chat_conversation_agent = agents.chat_conversation_agent()

    # Todo: Create tasks
    chat_conversation_task = tasks.chat_conversation_task(agent=chat_conversation_agent,
                                                          user_input=user_input)

    # Todo: Assemble the crew
    crew = Crew(
        agents=[chat_conversation_agent],
        tasks=[chat_conversation_task],
        process=Process.sequential,
        memory=True,
        verbose=True
    )

    # Todo: Run Crew
    if kick_off == True:
        result = crew.kickoff()
        # print('\n\n')
        # print('-------------------------------')
        # print("result: \n")
        # print(result)
        # print('\n\n')
        # print('-------------------------------')

        # print(f"""
        #         Task completed!
        #         Task: {research_task.output.description}
        #         Output: {research_task.output.raw}
        #     """)
        #
        # print(f"""
        #         Task completed!
        #         Task: {industry_analysis_task.output.description}
        #         Output: {industry_analysis_task.output.raw}
        #     """)
        #
        # print(f"""
        #             Task completed!
        #             Task: {meeting_strategy_task.output.description}
        #             Output: {meeting_strategy_task.output.raw}
        #         """)
        #
        # print(f"""
        #             Task completed!
        #             Task: {summary_and_briefing_task.output.description}
        #             Output: {summary_and_briefing_task.output.raw}
        #         """)

        # print(f"""
        #             Task completed!
        #             Task: {chat_conversation_task.output.description}
        #             Output: {chat_conversation_task.output.raw}
        #         """)

        return result


def append_to_file(file_path, text):
    try:
        with open(file_path, 'a') as file:
            file.write(text + "\n")  # Adds the string and a newline to the file
        print("Text successfully appended to the file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    load_dotenv()
    email_output_file = 'email_draft.txt'
    print("## Welcome to my Chat Application")

    while True:
        print('-------------------------------')
        user_input = input("Go ahead, Ask your question: ")

        if user_input == 'q':
            break
        result = main(user_input=user_input, kick_off=True)
        print(result)

        append_to_file(email_output_file, str(result))

        # f = open(email_output_file, "a")
        # f.write(str(result))
        # f.close()
