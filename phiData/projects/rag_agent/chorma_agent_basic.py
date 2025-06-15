import typer
from rich.prompt import Prompt
from typing import Optional, List
from phi.embedder.openai import OpenAIEmbedder
from phi.agent import Agent
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader
# from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.chroma import ChromaDb
from phi.storage.agent.json import JsonFileAgentStorage

from dotenv import load_dotenv

load_dotenv()

# Embedding Model
embeddings = OpenAIEmbedder()

# PDF Knowledge Base
# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["/Users/shubhamrathod/PycharmProjects/phiData/projects/rag_agent/ThaiRecipes.pdf"],
#     vector_db=ChromaDb(collection="recipes",
#                        embedder=embeddings),
# )
knowledge_base = PDFKnowledgeBase(
    path="/Users/shubhamrathod/PycharmProjects/phiData/projects/rag_agent/ThaiRecipes.pdf",
    vector_db=ChromaDb(collection="recipes",
                       embedder=embeddings),
    reader=PDFReader(chunk=True),
)

# Add information to the knowledge base
# Comment out after first run
knowledge_base.load(recreate=False)

# Add storage
storage = JsonFileAgentStorage(dir_path="tmp/agent_sessions_json")

# Agentic RAG
def pdf_agent(user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Add a tool to search the knowledge base which enables agentic RAG.
        search_knowledge=True,
        use_tools=True,
        show_tool_calls=True,
        # Add a tool to read chat history.
        read_chat_history=True,
        debug_mode=True,
    )
    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)

    # Run in cli
    # agent.cli_app(markdown=True)


if __name__ == "__main__":
    typer.run(pdf_agent)
