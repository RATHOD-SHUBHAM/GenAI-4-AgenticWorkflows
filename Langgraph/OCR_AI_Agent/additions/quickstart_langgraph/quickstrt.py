"""
https://docs.langchain.com/oss/python/langgraph/quickstart#full-code-example
1. Define tools and model
"""

from langchain.chat_models import init_chat_model
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv('AZURE_OPENAI_ENDPOINT')

model = init_chat_model("azure_openai:gpt-4o-mini", 
        temperature=0, 
        api_version='2025-03-01-preview'
        )

from langchain.tools import tool
# Define tools
@tool
def add(a:int, b:int) -> int:
    """
    Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a+b

@tool
def subtract(a:int, b:int) -> int:
    """Subtracts `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a-b

@tool
def multiply(a:int, b:int) -> int:
    """
    Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b

@tool
def divide(a:int, b:int) -> int:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with the tools
tools = [add, subtract, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)

"""
2. Define state
The graph’s state is used to store the messages and the number of LLM calls.
State in LangGraph persists throughout the agent’s execution.
The Annotated type with operator.add ensures that new messages are appended to the existing list rather than replacing it.
"""
from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add] # Store messages
    llm_calls : int # No of llm calls


"""
3. Define model node
The model node is used to call the LLM and decide whether to call a tool or not.
"""
from langchain.messages import SystemMessage

def llm_call(state:dict):
    """LLM decides whether to call a tool or not"""

    return {
        "messages" : [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls" : state.get('llm_calls', 0) + 1
    }

"""
4. Define tool node
The tool node is used to call the tools and return the results.
"""
from langchain.messages import ToolMessage

def tool_node(state: dict):
    """Performs the tool call"""
    result = []

    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])

        result.append(ToolMessage(content=observation, tool_call_id = tool_call["id"]))

    return {"messages" : result}

"""
5. Define end logic
The conditional edge function is used to route to the tool node or end based upon whether the LLM made a tool call.
"""
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"
    
    # Otherwise, we stop (reply to the user)
    return END


"""
6. Build and compile the agent
The agent is built using the StateGraph class and compiled using the compile method.
"""
# Build Workflow
agent_builder = StateGraph(MessagesState)

# Add Nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
    )
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
graph = agent.get_graph(xray=True)
png_bytes = graph.draw_mermaid_png()

# Save image to a file
with open("agent_graph.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as agent_graph.png")

os.system("open agent_graph.png")  # macOS
    
# Invoke
from langchain.messages import HumanMessage
messages = [HumanMessage(content="Add 3 and 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()