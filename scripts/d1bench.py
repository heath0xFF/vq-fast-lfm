#!/usr/bin/env python3
"""d1-3B /v1/systemone benchmark client (stdlib only; run on the VQ next to the server).
  ref  : the 3 reference sensor states + red-square image from the hackathon repo (fp32 transformers reference values)
  lat  : N distinct sensor states (cache-miss workload), sequential or concurrent -> latency p50/p95, decisions/s
"""
import argparse, base64, json, random, struct, sys, time, urllib.request, zlib, concurrent.futures as cf

Q = {
 "alert": {"type": "noul", "instructions": "Does this state need a human alert right now?"},
 "category": {"type": "choice", "instructions": "Which category fits this state best?",
              "criteria": {"thermal": "Temperature or heat problem", "motion": "Unexpected movement", "normal": "Nothing notable"}},
 "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["Low", "Moderate", "Critical"]},
}
REF_STATES = [
 ({"temp_c": 71.5, "temp_c_1min_ago": 58.0, "motion": "none"}, 0.6983, "thermal", 0.7522, 1.0),
 ({"temp_c": 36.8, "temp_c_1min_ago": 36.6, "motion": "none"}, 0.0686, "normal", 0.7315, 0.264),
 ({"temp_c": 37.0, "temp_c_1min_ago": 36.9, "motion": "sudden 2.1 g spike"}, 0.5998, "motion", 0.6362, 1.031),
]
REF_IMG_P = 0.9865

def png_red_square(n=256):
    rows = bytearray()
    for y in range(n):
        rows.append(0)
        for x in range(n):
            rows += bytes((220, 30, 30)) if 64 <= x <= 192 and 64 <= y <= 192 else bytes((255, 255, 255))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", n, n, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b"")

def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out, (time.perf_counter() - t) * 1000

def rand_state(rng):
    hot = rng.random() < 0.3
    t = rng.uniform(60, 95) if hot else rng.uniform(30, 42)
    return {"temp_c": round(t, 1), "temp_c_1min_ago": round(t - rng.uniform(-1, 15 if hot else 1), 1),
            "motion": rng.choice(["none", "none", "none", "sudden %.1f g spike" % rng.uniform(1, 4), "slow drift"]),
            "seq": rng.randint(0, 10**6)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "lat", "img"])
    ap.add_argument("--url", default="http://127.0.0.1:8098")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--split", action="store_true", help="send each question as its own concurrent request (per state)")
    a = ap.parse_args()
    if a.mode == "ref":
        post(a.url, {"state": json.dumps(REF_STATES[0][0]), "questions": Q})  # warm
        print(f"{'state':<6}{'p_alert':>9}{'(ref)':>9}{'dP':>8} {'category':<9}{'p(cat)':>7}{'(ref)':>7} {'sev':>6}{'(ref)':>7}{'ms':>8}")
        worst = 0
        for i, (st, p, cat, cf_, sev) in enumerate(REF_STATES):
            o, ms = post(a.url, {"state": json.dumps(st), "questions": Q}); A = o["answers"]
            pa, c = A["alert"]["noul"], A["category"]
            d = abs(pa - p); worst = max(worst, d, abs(A["severity"]["score"] - sev) / 2)
            print(f"{i:<6}{pa:9.4f}{p:9.4f}{pa-p:+8.4f} {c['choice']:<9}{c['probabilities'][c['choice']]:7.3f}{cf_:7.3f} {A['severity']['score']:6.3f}{sev:7.3f}{ms:8.0f}   {'OK' if c['choice']==cat else 'CATEGORY MISMATCH'}")
        print("max |dp| (alert, severity/2):", round(worst, 4))
    elif a.mode == "img":
        body = {"state": None, "images": ["data:image/png;base64," + base64.b64encode(png_red_square()).decode()],
                "questions": {"red": {"type": "noul", "instructions": "Is there a red object in this picture?"}}}
        for _ in range(a.warm): post(a.url, body)
        ms = []
        for _ in range(a.n):
            o, t = post(a.url, body); ms.append(t)
        ms.sort(); p = o["answers"]["red"]["noul"]
        print(f"image: p_red={p:.4f} (ref {REF_IMG_P})  p50={ms[len(ms)//2]:.0f} ms  min={ms[0]:.0f}  input_tokens={o['usage']['input_tokens']}")
    else:
        rng = random.Random(1)
        for _ in range(a.warm): post(a.url, {"state": json.dumps(rand_state(rng)), "questions": Q})
        states = [rand_state(rng) for _ in range(a.n)]
        t0 = time.perf_counter(); lat = []; toks = 0
        if a.split:
            def one(s):
                t = time.perf_counter()
                with cf.ThreadPoolExecutor(len(Q)) as ex2:
                    outs = list(ex2.map(lambda kv: post(a.url, {"state": json.dumps(s), "questions": {kv[0]: kv[1]}})[0], Q.items()))
                return sum(o["usage"]["input_tokens"] for o in outs), (time.perf_counter() - t) * 1000
            with cf.ThreadPoolExecutor(a.conc) as ex:
                for tk, ms in ex.map(one, states):
                    lat.append(ms); toks += tk
        else:
          with cf.ThreadPoolExecutor(a.conc) as ex:
            for o, ms in ex.map(lambda s: post(a.url, {"state": json.dumps(s), "questions": Q}), states):
                lat.append(ms); toks += o["usage"]["input_tokens"]
        wall = time.perf_counter() - t0; lat.sort()
        print(f"n={a.n} conc={a.conc}: p50={lat[len(lat)//2]:.0f} ms  p95={lat[int(len(lat)*0.95)-1]:.0f} ms  throughput={a.n/wall:.2f} decisions/s  ({toks/wall:.0f} input tok/s, {toks/a.n:.0f} tok/decision)")
if __name__ == '__main__':
    main()
