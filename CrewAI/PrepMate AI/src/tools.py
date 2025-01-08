import os
from dotenv import load_dotenv
import sendgrid
import base64
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition
from crewai.tools import tool
from crewai.tools import BaseTool
from pydantic import Field
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

# initialize SendGrid
api_key = os.environ.get("SENDGRID_API_KEY")
sg = sendgrid.SendGridAPIClient(api_key=api_key)


# Todo: Custom tools
class SendEmailTool(BaseTool):
    name: str = "Send Email"
    description: str = "Useful when it is needed to send emails."

    def _run(self, message_content: str)-> str:
        from_email = Email("smglab01@gmail.com")
        to = To("smglab01@gmail.com")
        subject = "Meeting Strategy"
        content = Content("text/html", message_content)

        mail = Mail(from_email, to, subject, content)

        # Attach file
        attachment_path = "email_draft.txt"
        with open(attachment_path, "rb") as f:
            data = f.read()
            f.close()

        encoded_file = base64.b64encode(data).decode()

        attached_file = Attachment(
            FileContent(encoded_file),
            FileName("my_docs.txt"),
            FileType("application/pdf"),
            Disposition("attachment")
        )

        mail.attachment = attached_file

        mail_json = mail.get()

        response = sg.client.mail.send.post(request_body=mail_json)
        print(response.status_code)
        print(response.headers)

        return "email_sent"



# Todo: Built In Tool
# Duck Duck Go Tools
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
send_email = SendEmailTool()
