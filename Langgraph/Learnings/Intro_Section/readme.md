# Langgraph

[Docs](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

---

When defining a graph, the first step is to define its State. </br>
The State includes the graph's schema and reducer functions that handle state updates. </br>
In our example, State is a TypedDict with one key: messages. The add_messages reducer function is used to append new messages to the list instead of overwriting it. Keys without a reducer annotation will overwrite previous values.

Step 1:
  * Start by creating a StateGraph.
  * A StateGraph object defines the structure of our chatbot as a "state machine". We'll add **nodes** to represent the llm and functions our chatbot can call and **edges** to specify how the bot should transition between these functions.

Our graph can now handle two key tasks:
  1. Each node can receive the current State as input and output an update to the state.
  2. Updates to messages will be appended to the existing list rather than overwriting it, thanks to the prebuilt add_messages function used with the Annotated syntax.

Step 2:
  * Next, add a "chatbot" node. Nodes represent units of work. They are typically regular Python functions.
  * Notice how the chatbot node function takes the current State as input and returns a dictionary containing an updated messages list under the key "messages". This is the basic pattern for all LangGraph node functions.

The add_messages function in our State will append the llm's response messages to whatever messages are already in the state.

Step 3:
  * Next, add an entry point. This tells our graph where to start its work each time we run it.

Step 4:
  * Similarly, set a finish point. This instructs the graph "any time this node is run, you can exit."

Step 5
  * Finally, we'll want to be able to run our graph. To do so, call "compile()" on the graph builder. This creates a "CompiledGraph" we can use invoke on our state.

We can visualize the graph using the get_graph method and one of the "draw" methods, like draw_ascii or draw_png. The draw methods each require additional dependencies.

---
