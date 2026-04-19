#!/usr/bin/env python3
import json
import re
from pathlib import Path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception:
                yield line_no, None


def infer_vuln_id_from_user(content: str) -> str:
    """Pull V-###### from the GroupID line in build_user() output."""
    for line in content.splitlines():
        line = line.strip()
        if line.lower().startswith("groupid"):
            m = re.search(r"V-\d{4,6}", line)
            if m:
                return m.group(0)
    return ""


def audit_file(path: Path):
    total_auto      = 0
    ok              = 0
    bad_empty       = []
    bad_no_logic    = []
    bad_markdown    = []
    bad_has_func    = []  # snippet accidentally contains a function declaration

    for line_no, rec in load_jsonl(path):
        if rec is None:
            continue
        messages = rec.get("messages") or rec.get("conversations") or []
        if not messages:
            continue

        user      = next((m for m in messages if m.get("role") == "user"),      None)
        assistant = next((m for m in messages if m.get("role") == "assistant"), None)
        if not user or not assistant:
            continue

        try:
            a_data = json.loads(assistant.get("content", ""))
        except Exception:
            continue

        if not isinstance(a_data, dict):
            continue

        classification = a_data.get("classification")
        ps             = a_data.get("powershell")

        if classification != "AUTOMATABLE":
            continue

        total_auto += 1

        if not isinstance(ps, str) or not ps.strip():
            bad_empty.append(line_no)
            continue

        # Should NOT contain a Function declaration — that's boilerplate territory
        if re.search(r"(?im)^\s*function\s+Get-V\d+\b", ps):
            bad_has_func.append(line_no)
            continue

        # Must contain actual logic
        if not re.search(r"(?im)^\s*\$\w+\s*=|\bIf\s*\(", ps):
            bad_no_logic.append(line_no)
            continue

        # No markdown fences
        if "```" in ps:
            bad_markdown.append(line_no)
            continue

        ok += 1

    print(f"\nFile: {path}")
    print(f"  AUTOMATABLE examples:      {total_auto}")
    print(f"  Passed semantic checks:    {ok}")
    print(f"  Empty snippet:             {len(bad_empty)}")
    print(f"  No logic found:            {len(bad_no_logic)}")
    if bad_no_logic:
        print(f"    lines: {bad_no_logic[:20]}")
    print(f"  Markdown fences:           {len(bad_markdown)}")
    if bad_markdown:
        print(f"    lines: {bad_markdown[:20]}")
    print(f"  Accidental func declaration: {len(bad_has_func)}")
    if bad_has_func:
        print(f"    lines: {bad_has_func[:20]}")


if __name__ == "__main__":
    base = Path("splits_final")
    for name in ["final_train_integrated.jsonl", "final_val_integrated.jsonl", "final_test_integrated.jsonl"]:
        path = base / name
        if path.exists():
            audit_file(path)
        else:
            print(f"Skipping (not found): {path}")