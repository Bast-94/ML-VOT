import requests

from src.yaml_utils import read_yaml

# Replace these variables with your actual values
config = read_yaml("secretdir/config.yml")
TOKEN = config["TOKEN"]
OWNER = config["OWNER"]
REPO = config["REPO"]

url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/artifacts"

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}

response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    # Do something with the response data, for example, print it
    resp = response.json()
    artifacts = resp["artifacts"]
    print([artifact["id"] for artifact in artifacts])
    
else:
    # Print the error message if the request was not successful
    print(f"Error: {response.status_code} - {response.text}")
