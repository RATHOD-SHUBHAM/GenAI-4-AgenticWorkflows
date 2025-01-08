import os
from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# Todo: Langchain tool
'''https://docs.crewai.com/concepts/langchain-tools'''


# search = DuckDuckGoSearchRun()

class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
    search: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"


search_tool = SearchTool()
