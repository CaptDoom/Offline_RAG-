import requests
import sys

def pull_model(name):
    print(f"Pulling {name}...")
    url = 'http://127.0.0.1:11434/api/pull'
    with requests.post(url, json={"name": name}, stream=True) as r:
        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text}")
            return
        for line in r.iter_lines():
            if line:
                print(line.decode('utf-8'))

if __name__ == "__main__":
    pull_model("llama3.2:1b")
