"""Publish the demo as a Hugging Face Space (Docker SDK, cpu-basic hardware; needs a PRO account since July 2026). Run on the Mac after `hf auth login`.

    python scripts/publish_space.py              # build space/ and show what would be uploaded
    python scripts/publish_space.py --upload     # create or update <user>/whisper-small-edacc-demo (public)

The Space gets the Dockerfile, exactly the files the image uses (the .dockerignore whitelist) and a README with
the Space settings (docs/space_readme.md). It is built only from a clean commit that is already pushed to GitHub,
and that commit is written into the README, so the Space can always be traced back to the code.
"""
import argparse
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPACE_NAME = "whisper-small-edacc-demo"
FILES = ["Dockerfile", "requirements-serve.lock", "configs/decode.yaml", "src/__init__.py", "src/asr.py",
         "src/serve/__init__.py", "src/serve/app.py", "src/serve/audio.py", "src/serve/static/index.html"]
CPU_THREADS = "2"  # cpu-basic has 2 vCPUs; more threads than CPUs only adds contention


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def clean_pushed_commit():
    if git("status", "--porcelain"):
        raise SystemExit("uncommitted changes: commit and push first, so the Space matches a real commit.")
    head = git("rev-parse", "HEAD")
    if head != git("rev-parse", "@{u}"):
        raise SystemExit("this commit is not pushed to GitHub yet: run git push first.")
    return head


def space_host(user, name):
    return f"{user}-{name}".lower().replace("_", "-").replace(".", "-") + ".hf.space"


def build(out, user, commit):
    shutil.rmtree(out, ignore_errors=True)
    for f in FILES:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, out / f)
    readme = (ROOT / "docs" / "space_readme.md").read_text()
    for k, v in {"USER": user, "HOST": space_host(user, SPACE_NAME), "COMMIT": commit[:7]}.items():
        readme = readme.replace("{" + k + "}", v)
    (out / "README.md").write_text(readme)
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    commit = clean_pushed_commit()
    api = HfApi()
    user = api.whoami()["name"]
    repo = f"{user}/{SPACE_NAME}"
    files = build(ROOT / "space", user, commit)
    print(f"space/ built from commit {commit[:7]}: {files}")
    if not args.upload:
        print(f"\nNot uploaded. Read space/README.md, then run again with --upload to publish {repo} (public).")
        return
    api.create_repo(repo, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    api.add_space_variable(repo, "CPU_THREADS", CPU_THREADS, description="CTranslate2 threads = vCPUs")
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(ROOT / "space"),
                      commit_message=f"Deploy github commit {commit[:7]}")
    print(f"uploaded. Build log and app: https://huggingface.co/spaces/{repo}")
    print(f"direct URL once running:   https://{space_host(user, SPACE_NAME)}")


if __name__ == "__main__":
    main()
