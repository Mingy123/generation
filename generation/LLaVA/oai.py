import os
import requests
import base64

# Configuration
API_KEY = os.getenv("OPENAI_AZURE_KEY")
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

def ask_question(text):
    payload = {
      "messages": [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": "You are an AI assistant that helps people find information. Follow all instructions carefully and strictly."
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text
            }
          ]
        }
      ],
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 4096
    }
    return req_api(payload)

def req_api(payload):
    ENDPOINT = "https://gpt4-hacx.openai.azure.com/openai/deployments/gpt-maybe/chat/completions?api-version=2024-02-15-preview"

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    return response.json()

if __name__ == "__main__":
    res = ask_question("Who is the first president of the united states?")
    print(type(res))
    print(res['choices'][0])
