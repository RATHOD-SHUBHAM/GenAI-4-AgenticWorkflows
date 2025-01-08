import smtplib

# list of email_id to send the mail
li = ["smglab01@gmail.com", "laravelshubham@gmail.com"]

for dest in li:
    s = smtplib.SMTP('smglab01@gmail.com.gmail.com', 587)
    s.starttls()
    s.login("sender_email_id", "sender_email_id_password")
    message = "Message_you_need_to_send"
    s.sendmail("sender_email_id", dest, message)
    s.quit()