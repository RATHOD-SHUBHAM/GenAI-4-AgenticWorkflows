from dotenv import load_dotenv
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()


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


# todo: use the tool
search_tool = SearchTool()
