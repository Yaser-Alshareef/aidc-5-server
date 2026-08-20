"""Submit our team's model to the board. Standard library only."""

import json, urllib.request, urllib.error

BOARD = "https://aidc.nadir.sh/model"

TEAM = "5"
BY = "Yaser Alshareef And Amr Algamidi"
IMAGE = "ghcr.io/yaser-alshareef/aidc-5-server:latest"


def request(url, body=None):
    data = json.dumps(body).encode() if body else None
    headers = {"User-Agent": "aidc-student/1.0"}
    if body:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


status, result = request("http://localhost:8000/generate")
print("my server said", status, result)
if status != 200:
    raise SystemExit("/generate route not working")

shared = result["shared"]

status, reply = request(BOARD, {
    "team": TEAM,
    "by": BY,
    "model": result["model"],
    "image": IMAGE,
    "tokens_per_sec": shared["tokens_per_sec"],
    "sample": shared["sample"],
})
print("the board said", status)
print(json.dumps(reply, indent=2))
