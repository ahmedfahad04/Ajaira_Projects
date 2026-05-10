import requests

BASE_URL = "http://localhost:4141"

data = '''
You are generating a probability-weighted distribution of Python solutions.

Given the original code below, generate 5 independent Python solutions to the same problem.
Each solution must be correct and executable. Think about the full space of ways this problem could be solved.

For each solution, assign a probability (0.0 to 1.0) representing how likely this 
approach would appear across the full distribution of valid solutions to this problem.
Probabilities do not need to sum to 1.0 across your 5 samples.

Output each variant within <response> tags containing:
- <probability>: float between 0.0 and 1.0
- <code>: complete, executable Python code only

Do not explain. Do not add comments describing what changed. 
Output only the 5 <response> blocks.

Original Code:
```python
import logging
import datetime


class AccessGatewayFilter:

    def __init__(self):
        pass

    def filter(self, request):
        request_uri = request['path']
        method = request['method']

        if self.is_start_with(request_uri):
            return True

        try:
            token = self.get_jwt_user(request)
            user = token['user']
            if user['level'] > 2:
                self.set_current_user_info_and_log(user)
                return True
        except:
            return False

    def is_start_with(self, request_uri):
        start_with = ["/api", '/login']
        for s in start_with:
            if request_uri.startswith(s):
                return True
        return False

    def get_jwt_user(self, request):
        token = request['headers']['Authorization']
        user = token['user']
        if token['jwt'].startswith(user['name']):
            jwt_str_date = token['jwt'].split(user['name'])[1]
            jwt_date = datetime.datetime.strptime(jwt_str_date, "%Y-%m-%d")
            if datetime.datetime.today() - jwt_date >= datetime.timedelta(days=3):
                return None
        return token

    def set_current_user_info_and_log(self, user):
        host = user['address']
        logging.log(msg=user['name'] + host + str(datetime.datetime.now()), level=1)
```
'''

def chat_completions(prompt):
    """
    OpenAI-style API
    Endpoint: /v1/chat/completions
    """

    url = f"{BASE_URL}/v1/chat/completions"

    payload = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    data = response.json()

    return data["choices"][0]["message"]["content"]


def messages_api(prompt):
    """
    Anthropic-style API
    Endpoint: /v1/messages
    """

    url = f"{BASE_URL}/v1/messages"

    payload = {
        "model": "gpt-5-mini",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    response = requests.post(
        url,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    data = response.json()

    return data["content"][0]["text"]


if __name__ == "__main__":

    prompt = data

    print("\nChoose API:")
    print("1 -> /v1/chat/completions")
    print("2 -> /v1/messages")

    choice = input("Enter choice: ")

    try:
        if choice == "1":
            reply = chat_completions(prompt)

        elif choice == "2":
            reply = messages_api(prompt)

        else:
            print("Invalid choice")
            exit()

        print("\nAssistant Response:\n")
        print(reply)

    except Exception as e:
        print("Error:", e)