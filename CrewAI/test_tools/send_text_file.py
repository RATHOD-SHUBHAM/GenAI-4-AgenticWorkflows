import os
from dotenv import load_dotenv
import sendgrid
from sendgrid import SendGridAPIClient
import base64
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition

load_dotenv()

api_key = os.environ.get("SENDGRID_API_KEY")
sg = sendgrid.SendGridAPIClient(api_key=api_key)

from_email = Email("smglab01@gmail.com")
to = To("smglab01@gmail.com")
subject = "Meeting Strategy"
content = Content("text/html", "HI")

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