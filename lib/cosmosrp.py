import requests

url = "https://api.pawan.krd/cosmosrp/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "cosmosrp",
    "messages": [
        {"role": "system", "content": "You are a fantasy world dungeon master."},
        {"role": "user", "content": "Describe the entrance of the ancient cave."},
    ],
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
