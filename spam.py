import imaplib
import email

mail=imaplib.IMAP4_SSL("imap.gmail.com")
mail.login("diwakar142008@gmail.com","vvlk tsvo brqy zwvl")

mail.select("inbox")
status,message=mail.search(None,"ALL")
email_ids=message[0].split()
print("total Email:",len(email_ids))
latest_email_id=email_ids[-1]
#fetech the last mail
status,data=mail.fetch(latest_email_id,"(RFC822)")
raw_email=data[0][1]
MSG=email.message_from_bytes(raw_email)
print("Subject:",MSG["subject"])
print("From:",MSG["from"])
