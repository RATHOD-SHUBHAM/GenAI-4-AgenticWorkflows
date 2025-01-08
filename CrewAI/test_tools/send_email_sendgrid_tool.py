import os
from dotenv import load_dotenv
import sendgrid
from sendgrid import SendGridAPIClient
import base64
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition

load_dotenv()


# Todo: Test 1
#
# message = Mail(
#     from_email='smglab01@gmail.com',
#     to_emails='smglab01@gmail.com',
#     subject='Testing this tool i built',
#     html_content='<strong>Dont panic, you know me</strong>')
#
# sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
# response = sg.send(message)
# print(response.status_code, response.body, response.headers)

# Todo: Test 2

# import os
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail

# message = Mail(
#     from_email='smglab01@gmail.com',
#     to_emails='smglab01@gmail.com',
#     subject='Testing this tool i built',
#     html_content='<strong>Dont panic, you know me</strong>')
# try:
#     sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
#     response = sg.send(message)
#     print(response.status_code)
#     print(response.body)
#     print(response.headers)
# except Exception as e:
#     print(e.message)


# Todo: Test 3
api_key = os.environ.get("SENDGRID_API_KEY")
sg = sendgrid.SendGridAPIClient(api_key=api_key)

from_email = Email("smglab01@gmail.com")
to = To("smglab01@gmail.com")
subject = "Dummy Attachment"
content = Content("text/html", "Dont panic, you know me!</b>")
mail = Mail(from_email, to, subject, content)

with open('cv.pdf', "rb") as f:
  data = f.read()
  f.close()

encoded_file = base64.b64encode(data).decode()

attached_file = Attachment(
  FileContent(encoded_file),
  FileName("cv.pdf"),
  FileType("application/pdf"),
  Disposition("attachment")
)

mail.attachment = attached_file

mail_json = mail.get()

response = sg.client.mail.send.post(request_body=mail_json)
print(response.status_code)
print(response.headers)