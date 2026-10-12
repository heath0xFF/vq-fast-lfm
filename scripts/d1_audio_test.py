import base64, json, time, urllib.request, sys
b = base64.b64encode(open("jfk.wav","rb").read()).decode()
body = {"state": "Recording from a public event.", "files": ["data:audio/wav;base64," + b],
 "questions": {"kind": {"type": "choice", "instructions": "What kind of utterance is this?", "criteria": {"request": "A request to do something", "question": "A question asking for information", "speech": "A speech to an audience"}},
               "calm": {"type": "score", "instructions": "How calm is the speaker?", "criteria": ["Agitated", "Neutral", "Calm"]}}}
ms = []
for i in range(4):
    t = time.perf_counter(); req = urllib.request.Request("http://127.0.0.1:8098/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    o = json.load(urllib.request.urlopen(req, timeout=300)); ms.append((time.perf_counter() - t) * 1000)
a = o["answers"]
print("audio: kind=%s p=%s calm=%.2f  input_tokens=%s  latency ms: %s" % (a["kind"]["choice"], {k: round(v, 3) for k, v in a["kind"]["probabilities"].items()}, a["calm"]["score"], o["usage"]["input_tokens"], [round(x) for x in ms]))
