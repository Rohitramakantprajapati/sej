"""Lightweight end-to-end smoke test for AutoAnalyst.

This script checks the live API, uploads a sample dataset, retrains the model,
and verifies the rows/session endpoints without requiring third-party test tools.
"""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path


DEFAULT_BASE_URL = "http://localhost:8000"


def request_json(base_url: str, path: str, method: str = "GET", data: bytes | None = None, headers: dict | None = None) -> dict:
    request_headers = {"Accept": "application/json"}
    if headers:
      request_headers.update(headers)
    request = urllib.request.Request(f"{base_url}{path}", data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed with HTTP {error.code}: {body}") from error


def build_multipart(field_name: str, filename: str, content: bytes, content_type: str = "text/csv") -> tuple[bytes, str]:
    boundary = f"----AutoAnalystBoundary{uuid.uuid4().hex}"
    buffer = io.BytesIO()
    buffer.write(f"--{boundary}\r\n".encode())
    buffer.write(
        (
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
    )
    buffer.write(content)
    buffer.write(f"\r\n--{boundary}--\r\n".encode())
    return buffer.getvalue(), f"multipart/form-data; boundary={boundary}"


def make_sample_csv(path: Path) -> None:
    path.write_text(
        "region,sales,profit,target\n"
        "North,120,35,1\n"
        "South,180,52,0\n"
        "East,95,24,1\n"
        "West,260,88,0\n"
        "North,310,102,1\n"
        "South,150,41,0\n"
        "East,280,93,1\n"
        "West,175,57,0\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the AutoAnalyst backend.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL, default: http://localhost:8000")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Checking {base_url} ...")

    health = request_json(base_url, "/health")
    print(f"Health OK: {health.get('status')} ({health.get('environment')})")

    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "smoke_test.csv"
        make_sample_csv(csv_path)
        content = csv_path.read_bytes()
        body, content_type = build_multipart("file", csv_path.name, content, "text/csv")
        upload = request_json(base_url, "/upload", method="POST", data=body, headers={"Content-Type": content_type})

        session_id = upload.get("session_id")
        if not session_id:
            raise RuntimeError("Upload response did not return a session_id.")
        print(f"Upload OK: session_id={session_id}")

        model = request_json(
            base_url,
            "/model",
            method="POST",
            data=json.dumps({"session_id": session_id, "target_col": "target"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        task = model.get("task_info", {}).get("task", "unknown")
        print(f"Model OK: {task}")

        rows = request_json(base_url, f"/rows/{session_id}?limit=5&offset=0")
        print(f"Rows OK: {len(rows.get('rows', []))} sample row(s)")

        session = request_json(base_url, f"/sessions/{session_id}")
        print(f"Session OK: {len(session.get('columns', []))} columns")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())