"""Speed of the live Space, as fixed in docs/decisions.md (2026-10-28, before the Space existed). Run on the Mac.

    python scripts/measure_space.py latency /tmp/lat/*.aiff
    python scripts/measure_space.py cold-start

latency:    one warm-up request (not counted), then each clip --repeats times in a row. Server time is the
            processing_seconds the service reports (transcription only). Round trip is what a visitor waits for,
            including the upload from this machine and the network.
            The transcript is saved too (use synthetic or your own speech only, never dataset audio).
cold-start: restarts the Space through the API (needs `hf auth login`) and times until /health answers from the
            new container (its started_at changes). Includes Hugging Face scheduling and the container start.
Results go to results/serving/space_<mode>_<n>.json; an existing file is never overwritten.
"""
import argparse
import datetime
import json
import pathlib
import ssl
import statistics
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
URL = "https://mddalwai-whisper-small-edacc-demo.hf.space"
REPO = "mddalwai/whisper-small-edacc-demo"


def ssl_context():
    """The operating system's certificates (macOS Keychain) via truststore, which huggingface_hub already installs.
    Python from python.org on macOS ships without certificates, so plain urllib fails with CERTIFICATE_VERIFY_FAILED."""
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        return ssl.create_default_context()


CTX = ssl_context()


def call(path, data=None, timeout=120):
    req = urllib.request.Request(URL + path, data=data, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return json.loads(r.read())


def summarize(values):
    return {"median": round(statistics.median(values), 3), "max": round(max(values), 3), "n": len(values)}


def save(mode, result):
    out_dir = ROOT / "results" / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 1
    while (out_dir / f"space_{mode}_{n}.json").exists():
        n += 1
    path = out_dir / f"space_{mode}_{n}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {path.relative_to(ROOT)}")


def latency(args):
    health = call("/health")
    clips = [(pathlib.Path(p).name, pathlib.Path(p).read_bytes()) for p in args.clips]
    call("/transcribe", clips[0][1])  # warm-up, not counted
    rows = []
    print(f"{'clip':14} {'audio s':>8} {'server median/max s':>20} {'round trip median/max s':>24} words")
    for name, data in clips:
        server, round_trip = [], []
        try:
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                r = call("/transcribe", data)
                round_trip.append(time.perf_counter() - t0)
                server.append(r["processing_seconds"])
        except urllib.error.HTTPError as e:
            rows.append({"clip": name, "error": e.code, "detail": e.read().decode()[:200]})
            print(f"{name:14} error {e.code}")
            continue
        row = {"clip": name, "audio_seconds": r["audio_seconds"], "words": len(r["text"].split()), "text": r["text"],
               "server_seconds": summarize(server), "round_trip_seconds": summarize(round_trip)}
        rows.append(row)
        s, t = row["server_seconds"], row["round_trip_seconds"]
        print(f"{name:14} {r['audio_seconds']:8.1f} {s['median']:9.2f} / {s['max']:<9.2f} "
              f"{t['median']:11.2f} / {t['max']:<10.2f} {row['words']}")
    save("latency", {"date": datetime.datetime.now().isoformat(timespec="seconds"), "url": URL,
                     "health": health, "repeats": args.repeats, "clips": rows})


def cold_start(args):
    from huggingface_hub import HfApi

    api, rows = HfApi(), []
    for i in range(args.times):
        before = call("/health")["started_at"]
        t0 = time.time()
        api.restart_space(REPO)
        while True:
            if time.time() - t0 > args.timeout:
                raise SystemExit(f"no new container after {args.timeout} s")
            try:
                h = call("/health", timeout=10)
                if h.get("started_at") not in (None, before):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
                pass  # the Space answers with errors or a "starting" page while it restarts
            time.sleep(1)
        row = {"restart_to_ready_seconds": round(time.time() - t0, 1), "load_seconds": h["load_seconds"],
               "warmup_seconds": h["warmup_seconds"]}
        rows.append(row)
        print(f"restart {i + 1}: ready after {row['restart_to_ready_seconds']} s "
              f"(model load {row['load_seconds']} s, warm-up {row['warmup_seconds']} s)")
        time.sleep(args.pause)
    save("cold_start", {"date": datetime.datetime.now().isoformat(timespec="seconds"), "url": URL, "repo": REPO,
                        "restarts": rows,
                        "restart_to_ready_seconds": summarize([r["restart_to_ready_seconds"] for r in rows])})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    lat = sub.add_parser("latency")
    lat.add_argument("clips", nargs="+")
    lat.add_argument("--repeats", type=int, default=5)
    cold = sub.add_parser("cold-start")
    cold.add_argument("--times", type=int, default=3)
    cold.add_argument("--timeout", type=float, default=900)
    cold.add_argument("--pause", type=float, default=30)
    args = ap.parse_args()
    latency(args) if args.mode == "latency" else cold_start(args)


if __name__ == "__main__":
    main()
