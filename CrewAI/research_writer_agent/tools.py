import os
from dotenv import load_dotenv

from crewai_tools import (
    DirectoryReadTool,
    FileReadTool,
    WebsiteSearchTool
)
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()
os.environ['SERPAPI_API_KEY'] = os.getenv('SERPAPI_API_KEY')

# Todo: Crewai tool
web_rag_tool = WebsiteSearchTool()

# Todo: Langchain tool
'''https://docs.crewai.com/concepts/langchain-tools'''

SerpAPIWrapper()


class SearchTool(BaseTool):
    name: str = "Search"
    description: str = "Useful for search-based queries. Use this to find current information about markets, companies, and trends."
    search: SerpAPIWrapper = Field(default_factory=SerpAPIWrapper)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            return self.search.run(query)
        except Exception as e:
            return f"Error performing search: {str(e)}"


search = SearchTool()
