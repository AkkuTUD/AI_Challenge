import requests

url = "https://dps-challenge.netlify.app/.netlify/functions/api/challenge"
headers = {
    "Content-Type": "application/json"
}
data = {
    "github": "https://github.com/AkkuTUD/AI_Challenge",
    "email": "akanksha.tanwar@tu-dortmund.de",
    "url": "https://prediction-api-593253495834.europe-west3.run.app/",
    "notes": "I created the model using XGBoost algorithm and deployed the Flask API using Google cloud"
}

response = requests.post(url, json=data, headers=headers)

print(response.status_code)
print(response.json())