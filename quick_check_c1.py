import requests

API = "http://localhost:8000/chat"

history = [
    {"role": "user", "content": "We need a solution for senior leadership."},
]
r1 = requests.post(API, json={"messages": history}, timeout=60).json()
print("TURN 1:", r1["reply"][:150], r1["recommendations"], r1["end_of_conversation"])
history.append({"role": "assistant", "content": r1["reply"]})

history.append({"role": "user", "content": "The pool consists of CXOs, director-level postions; people with more than 15 years of experience."})
r2 = requests.post(API, json={"messages": history}, timeout=60).json()
print("TURN 2:", r2["reply"][:150], r2["recommendations"], r2["end_of_conversation"])
history.append({"role": "assistant", "content": r2["reply"]})

history.append({"role": "user", "content": "Selection — comparing candidates against a leadership benchmark."})
r3 = requests.post(API, json={"messages": history}, timeout=60).json()
print("TURN 3:", r3["reply"][:150], r3["recommendations"], r3["end_of_conversation"])
history.append({"role": "assistant", "content": r3["reply"]})

history.append({"role": "user", "content": "Perfect, that's what we need."})
r4 = requests.post(API, json={"messages": history}, timeout=60).json()
print("TURN 4:", r4["reply"][:150], r4["recommendations"], r4["end_of_conversation"])