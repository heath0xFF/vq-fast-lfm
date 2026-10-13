#!/usr/bin/env python3
"""d1 fidelity harness (stdlib). Sends a fixed, seeded set of decisions to a llama-server /v1/systemone.
  --save ref.json : store answers (use the F16 model as the reference)
  --ref  ref.json : compare to stored reference -> mean/max |dP| for noul, choice-prob L1, score error, top-1 agreement, 0.5-threshold flips
Dataset: 40 sensor states + 40 support tickets. Seed 1 is frozen for the saved references.
Different seeds can repeat tickets; use d1_calibration.py to check or filter calibration inputs.
"""
import argparse, json, random, statistics as st, urllib.request, time, sys
SENS_Q = {"alert": {"type": "noul", "instructions": "Does this state need a human alert right now?"},
 "category": {"type": "choice", "instructions": "Which category fits this state best?", "criteria": {"thermal": "Temperature or heat problem", "motion": "Unexpected movement", "normal": "Nothing notable"}},
 "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["Low", "Moderate", "Critical"]}}
TICK_Q = {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
 "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "fraud": "Suspected unauthorised use"}},
 "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "Today", "Blocking the customer now"]}}
def sensor(rng):
    hot = rng.random() < 0.4; t = rng.uniform(55, 95) if hot else rng.uniform(30, 42)
    return json.dumps({"temp_c": round(t, 1), "temp_c_1min_ago": round(t - rng.uniform(-2, 18 if hot else 1.5), 1),
        "motion": rng.choice(["none", "none", "sudden %.1f g spike" % rng.uniform(1, 4), "slow drift", "vibration"]), "door": rng.choice(["closed", "closed", "open"])})
A = ["I was charged twice this month", "My card was billed after I cancelled", "The app crashes when I open settings", "I cannot log in since yesterday",
     "There are purchases I did not make", "Please send me last year's invoices", "The website shows a 500 error at checkout", "Someone used my account from another country"]
B = ["please refund one of them.", "this is blocking my work right now.", "can you look at it when you have time?", "I am very unhappy and want my money back.",
     "it started today.", "nobody has replied for a week.", "thanks in advance.", "I need this fixed immediately."]
def ticket(rng): return "Customer message: " + rng.choice(A) + ", " + rng.choice(B)
def dataset(seed):
    rng = random.Random(seed)
    return [(sensor(rng), SENS_Q) for _ in range(40)] + [(ticket(rng), TICK_Q) for _ in range(40)]
def labels(state):
    """Rule-based ground truth for the synthetic set (None = ambiguous, excluded)."""
    try:
        d = json.loads(state)
        t = d["temp_c"]; spike = "sudden" in d["motion"]
        hot = t >= 55
        return {"alert": hot or spike, "category": "thermal" if hot else ("motion" if spike else "normal")}
    except Exception:
        pass
    a = state.split(": ", 1)[1]
    team = next((v for k, v in [("charged twice", "billing"), ("billed after I cancelled", "billing"), ("invoices", "billing"), ("app crashes", "technical"),
                 ("cannot log in", "technical"), ("500 error", "technical"), ("did not make", "fraud"), ("another country", "fraud")] if k in a), None)
    refund_txt = ("refund" in a) or ("money back" in a)
    out = {"team": team}
    if refund_txt: out["refund"] = True
    elif team in ("technical", "fraud"): out["refund"] = False
    return out
def accuracy(states, ans):
    n_a = c_a = n_c = c_c = 0; brier = []
    for (s, _), x in zip(states, ans):
        L = labels(s)
        for k, v in L.items():
            if v is None or k not in x: continue
            if k in ("alert", "refund"):
                n_a += 1; p = x[k]["noul"]; c_a += (p >= .5) == v; brier.append((p - float(v)) ** 2)
            else:
                n_c += 1; c_c += x[k]["choice"] == v
    return f"labels: noul acc={c_a}/{n_a} brier={st.mean(brier):.3f} | choice acc={c_c}/{n_c}"
def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)
def collect(url, seed):
    out = []; t = time.perf_counter()
    for s, q in dataset(seed): out.append(post(url, {"state": s, "questions": q})["answers"])
    return out, (time.perf_counter() - t) / 80 * 1000
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8097"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save"); ap.add_argument("--ref"); ap.add_argument("--label", default=""); ap.add_argument("--acc", action="store_true")
    a = ap.parse_args(); ans, ms = collect(a.url, a.seed)
    if a.acc: print(f"{a.label:<18} {accuracy(dataset(a.seed), ans)} | {ms:.0f} ms/decision")
    if a.save: json.dump(ans, open(a.save, "w")); print(f"saved {len(ans)} decisions ({ms:.0f} ms each)")
    if a.ref:
        ref = json.load(open(a.ref)); dn, dc, ds, agree, flips, n_c, n_n = [], [], [], 0, 0, 0, 0
        for x, r in zip(ans, ref):
            for k, q in x.items():
                rq = r[k]
                if q["type"] == "noul": dn.append(abs(q["noul"] - rq["noul"])); n_n += 1; flips += (q["noul"] >= .5) != (rq["noul"] >= .5)
                elif q["type"] == "choice":
                    ks = rq["probabilities"].keys(); dc.append(sum(abs(q["probabilities"][o] - rq["probabilities"][o]) for o in ks) / 2); n_c += 1; agree += q["choice"] == rq["choice"]
                else: ds.append(abs(q["score"] - rq["score"]) / (len(rq["probabilities"]) - 1))
        print(f"{a.label:<18} noul: mean|dP|={st.mean(dn):.4f} max={max(dn):.3f} flips@0.5={flips}/{n_n} | choice: TV={st.mean(dc):.4f} top1={agree}/{n_c} | score: mean|dLevel|/range={st.mean(ds):.4f} max={max(ds):.3f} | {ms:.0f} ms/decision")
if __name__ == '__main__':
    main()
