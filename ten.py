
import email
from operator import contains

def dectect_spam(email):   
    if email.sender=="promotions@online.com":
        return "spam"
    if contains(email.title,['tax','review','lottery']):
        return "spam"
    return 'good'
print(dectect_spam(email))    