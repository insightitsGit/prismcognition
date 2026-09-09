"""Build and upload PrismCognition to PyPI or TestPyPI.

Read the token from the environment only. Never pass it as a CLI argument
and never print it.

    set PYPI_API_TOKEN=pypi-...
    python tools/publish.py

    set PYPI_REPOSITORY=testpypi
    set TEST_PYPI_API_TOKEN=pypi-...
    python tools/publish.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
REPOSITORIES = {
    "pypi": "https://upload.pypi.org/legacy/",
    "testpypi": "https://test.pypi.org/legacy/",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and upload PrismCognition to PyPI.")
    parser.add_argument(
        "--repository",
        default=os.environ.get("PYPI_REPOSITORY", "pypi"),
        choices=sorted(REPOSITORIES),
        help="Destination index. Defaults to PYPI_REPOSITORY or pypi.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not fail if this version is already on the index.",
    )
    args = parser.parse_args(argv)

    token_key = "TEST_PYPI_API_TOKEN" if args.repository == "testpypi" else "PYPI_API_TOKEN"
    token = os.environ.get(token_key) or os.environ.get("TWINE_PASSWORD")
    if not token:
        print(
            f"Set {token_key} (or TWINE_PASSWORD) in the environment. "
            "Do not paste the token into chat or commit it.",
            file=sys.stderr,
        )
        return 2
    if not token.startswith("pypi-"):
        print(
            f"{token_key} does not look like a PyPI API token (expected pypi-...).",
            file=sys.stderr,
        )
        return 2

    if DIST.exists():
        shutil.rmtree(DIST)
    subprocess.check_call([sys.executable, "-m", "build"], cwd=ROOT)
    subprocess.check_call([sys.executable, "-m", "twine", "check", "--strict", * _dist_files()])

    env = os.environ.copy()
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    upload = [
        sys.executable,
        "-m",
        "twine",
        "upload",
        "--non-interactive",
        "--repository-url",
        REPOSITORIES[args.repository],
    ]
    if args.skip_existing:
        upload.append("--skip-existing")
    upload.extend(_dist_files())
    subprocess.check_call(upload, cwd=ROOT, env=env)
    print(f"Uploaded {_package_label()} to {args.repository}.")
    return 0


def _dist_files() -> list[str]:
    files = sorted(str(path) for path in DIST.glob("*") if path.suffix in {".whl", ".gz"})
    if not files:
        raise SystemExit("build produced no distribution files")
    return files


def _package_label() -> str:
    wheels = list(DIST.glob("*.whl"))
    return wheels[0].name if wheels else "distribution"


if __name__ == "__main__":
    raise SystemExit(main())
