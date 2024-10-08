import requests


url = "http://localhost:8000/predict"

test_data = {
  "goalamount": 76159,
  "raisedamount": 73518.02895458053,
  "durationdays": 15,
  "numbackers": 1903,
  "category": "film",
  "launchmonth": "september",
  "country": "australia",
  "currency": "gbp",
  "ownerexperience": 11,
  "videoincluded": "yes",
  "socialmediapresence": 41758,
  "numupdates": 3
}

response = requests.post(url=url, json=test_data)
print(f"Test case 1 Campaign Success Prediction: {response.json()}")