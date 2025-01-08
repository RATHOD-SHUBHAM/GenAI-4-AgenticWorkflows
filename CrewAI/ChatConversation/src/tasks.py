from textwrap import dedent
from crewai import Task
from ChatConversation.src.tools import search_tool, send_email




class ChatConversationalTask:
    # Todo: Add Tasks Here

    def chat_conversation_task(self, agent, user_input):
        return Task(
            description=dedent(f"""
                        Engage in a free-humanly-form conversation with the user based on their question. 
                        You should listen to the user’s question or statement, understand the context, and generate a relevant, friendly, and informative response.

                        You can use the available tools to gather additional information if needed, but the main goal is to maintain a natural and engaging dialogue. The agent should not be overly formal or robotic; instead, it should aim to sound conversational and approachable.
                        
                        Send an email when the user asks for it. If the user requests an email, use the available tools to send an email with the chat history as the context.
                        
                        User Question: {user_input}
                        """),
            expected_output=dedent(f"""
                        A friendly and relevant response to the user’s query or statement. 
                        You should maintain a conversational tone, provide any necessary context or information, and adapt its response to the user's tone or style where possible. 
                        If the question requires additional research, you may pull in external data or utilize its tools, but the response should feel natural and spontaneous.
                        If email was sent, just say email was sent and ask if the user needs any more assistance
                        """),
            name="conversation_task",
            agent=agent,
            tools=[search_tool, send_email],
            async_execution=True
        )