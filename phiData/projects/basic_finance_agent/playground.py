from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.playground import Playground, serve_playground_app
from dotenv import load_dotenv

load_dotenv()

llm = Groq(id="llama3-groq-70b-8192-tool-use-preview")

# Todo: Web search agent
web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=llm,
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Todo: Financial agent
finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=llm,
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

app = Playground(agents=[finance_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)
