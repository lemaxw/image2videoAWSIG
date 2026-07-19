"""Record human acceptance or rejection in a candidate result JSON file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def update_feedback(
    result_path: Path,
    *,
    status: str,
    rating: int | None = None,
    issues: Iterable[str] = (),
    notes: str = "",
) -> dict:
    if status not in {"accepted", "rejected", "pending"}:
        raise ValueError(f"Unsupported feedback status: {status}")
    if rating is not None and not 1 <= rating <= 5:
        raise ValueError("rating must be between 1 and 5")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    normalized_issues = list(dict.fromkeys(str(issue).strip() for issue in issues if str(issue).strip()))
    payload["human_feedback"] = {
        "status": status,
        "rating": rating,
        "issue_codes": normalized_issues,
        "notes": str(notes).strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["state"] = "ACCEPTED" if status == "accepted" else "REJECTED" if status == "rejected" else "HUMAN_REVIEW"
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(result_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", choices=("accept", "reject", "pending"))
    parser.add_argument("--result", type=Path, action="append", required=True, help="Candidate .result.json; repeat for several candidates")
    parser.add_argument("--rating", type=int)
    parser.add_argument("--issues", default="", help="Comma-separated normalized issue codes")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    status = {"accept": "accepted", "reject": "rejected", "pending": "pending"}[args.status]
    issues = [part.strip() for part in args.issues.split(",") if part.strip()]
    for path in args.result:
        updated = update_feedback(path, status=status, rating=args.rating, issues=issues, notes=args.notes)
        print(json.dumps({"result": str(path), "feedback": updated["human_feedback"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
