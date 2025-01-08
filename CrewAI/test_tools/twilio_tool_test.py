from langchain_community.utilities.twilio import TwilioAPIWrapper
from dotenv import load_dotenv

# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
load_dotenv()

# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
# account_sid = os.environ["TWILIO_ACCOUNT_SID"]
# auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client()
# client = Client(account_sid, auth_token)

message = client.messages.create(
    body="This is a message from me to me?",
    from_="+",
    to="+",
)

print(message.body)