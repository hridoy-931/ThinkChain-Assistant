import requests
import json

def query_ollama(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}
    response = requests.post(url, json=payload, stream=True)

    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                output += data["response"]
    return output
