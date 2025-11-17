# 6 Core Steps for Building Agents

1. Define tools and model

2. Define state
The graph’s state is used to store the messages and the number of LLM calls.

Build Nodes
3. Define model node
The model node is used to call the LLM and decide whether to call a tool or not.

4. Define tool node
The tool node is used to call the tools and return the results.

5. Define end logic
The conditional edge function is used to route to the tool node or end based upon whether the LLM made a tool call.

6. Build and compile the agent
The agent is built using the StateGraph class and compiled using the compile method.
    1. Build workflow - use StateGraph
    2. Add nodes
    3. Add edges to connect nodes
    4. Compile the agent
    5. Show the agent
    6. Invoke


# Activate the Virtual Environment
source .venv/bin/activate