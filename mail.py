import datetime
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import smtplib

start = time.time()
user = 'basilegarcia@gmail.com'
password = 'Mangas!33191994'
subject = ' 🦄 [GETZ-SERVER] Job <{}> is done! 🦄 '
recipient = 'basilegarcia@gmail.com'
body = 'The Job called "{}" and located in {} is now done.\nExecution time = {}'


def auto_send(job_name, main_file, attachment=None):
    DIR = os.path.dirname(os.path.abspath(main_file))
    exec_time = str(
        datetime.timedelta(seconds=int(time.time() - start))
    )

    while True:

        done = send_mail_using_g_mail(
            user=user,
            pwd=password,
            recipient=recipient,
            subject=subject.format(job_name),
            body=body.format(job_name, DIR, exec_time),
            attachment=attachment
        )

        if done:
            break


def send_mail_using_g_mail(user, pwd, recipient, subject, body, attachment):
    to = recipient if type(recipient) is list else [recipient]

    # Prepare actual message
    msg = MIMEMultipart()

    msg['From'] = user
    msg['To'] = recipient
    msg['Subject'] = subject

    body = body

    msg.attach(MIMEText(body, 'plain'))

    if attachment:
        with open(attachment, "rb") as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload((f).read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition', 'attachment', filename=os.path.basename(attachment)
            )

            msg.attach(part)

    message = msg.as_string()

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(user, to, message)
        server.close()
        return True

    except Exception as e:
        print(str(e))
        return False

