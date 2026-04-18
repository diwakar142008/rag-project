import requests

print("Simple AI chatbot (type 'exit' to quit)")
while True:
    user_input=input("you:")
    if user_input.lower=="exit":
        print("goodbye")
        break
    try:
        response=requests.post( "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": user_input,
                "stream": False
            }
        )
        result=response.json()
        print("AI:",result["response"])
    except Exception as e:
        print("Error:",e)        