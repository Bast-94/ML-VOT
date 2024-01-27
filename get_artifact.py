import requests

from src.parsers import get_git_manager_args
from src.yaml_utils import read_yml

# Replace these variables with your actual values
config = read_yml("secretdir/config.yml")
TOKEN = config["TOKEN"]
OWNER = config["OWNER"]
REPO = config["REPO"]

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}
args = get_git_manager_args()

if args.commands == "artifacts":
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/artifacts"

if args.commands == "tree":
    user = args.user if args.user != "" else OWNER
    repo = args.repo if args.repo != "" else REPO
    branch = args.branch if args.branch != "" else "main"
    url = f"https://api.github.com/repos/{user}/{repo}/git/trees/{branch}"

response = requests.get(url, headers=headers)

# Check the response
if response.status_code == 200:
    # Do something with the response data, for example, print it
    resp = response.json()
    print(resp)

else:
    # Print the error message if the request was not successful
    print(f"Error: {response.status_code} - {response.text}")
