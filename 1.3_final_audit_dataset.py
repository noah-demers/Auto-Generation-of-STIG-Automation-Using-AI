#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from lxml import etree


def _txt(node):
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def load_stig_mapping(path: Path):
    tree = etree.parse(str(path))
    root = tree.getroot()

    stig_by_content = {}
    stig_by_shortname = {}

    for stig in root.findall(".//STIG"):
        name         = _txt(stig.find("Name"))
        short_name   = _txt(stig.find("ShortName"))
        stig_content = _txt(stig.find("StigContent"))
        content_hash = _txt(stig.find("ContentHash"))
        ps_module    = _txt(stig.find("PsModule"))
        counts_node  = stig.find("Counts")

        cat_i = cat_ii = cat_iii = 0
        if counts_node is not None:
            text  = _txt(counts_node)
            m_i   = re.search(r"CATI(\d+)", text)
            m_ii  = re.search(r"CATII(\d+)", text)
            m_iii = re.search(r"CATIII(\d+)", text)
            cat_i   = int(m_i.group(1)) if m_i else 0
            cat_ii  = int(m_ii.group(1)) if m_ii else 0
            cat_iii = int(m_iii.group(1)) if m_iii else 0

        entry = {
            "name": name,
            "short_name": short_name,
            "stig_content": stig_content,
            "content_hash": content_hash,
            "ps_module": ps_module,
            "cat_i": cat_i,
            "cat_ii": cat_ii,
            "cat_iii": cat_iii,
        }

        if stig_content:
            stig_by_content[stig_content] = entry
        if short_name:
            stig_by_shortname[short_name] = entry

    return stig_by_content, stig_by_shortname


def infer_stig_shortname_from_user(msg_content: str) -> str:
    """
    Matches build_user() output which starts lines with:
      STIG: Win11
      GroupID (VulnID): V-243466
      ...
    """
    for line in msg_content.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("stig:") or \
           lower.startswith("stigshortname:") or \
           lower.startswith("shortname:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()

    first_line = msg_content.splitlines()[0].strip() if msg_content.splitlines() else ""
    if first_line and " " not in first_line and len(first_line) <= 20:
        return first_line

    return ""


def validate_assistant_payload(assistant_content: str):
    """
    Returns (ok: bool, reason: str, classification: str | None, vuln_id: str).
    """
    try:
        data = json.loads(assistant_content)
    except Exception:
        return False, "assistant_not_json", None, ""

    if not isinstance(data, dict):
        return False, "assistant_not_object", None, ""

    if "classification" not in data or "powershell" not in data:
        return False, "missing_keys", None, ""

    classification = data["classification"]
    powershell     = data["powershell"]

    if classification not in {"AUTOMATABLE", "MANUAL", "UNKNOWN"}:
        return False, f"bad_classification:{classification}", classification, ""

    if classification in {"MANUAL", "UNKNOWN"}:
        if powershell not in (None, ""):
            return False, "non_null_ps_for_manual", classification, ""
        return True, "ok", classification, ""

    if not isinstance(powershell, str) or not powershell.strip():
        return False, "empty_ps_for_automatable", classification, ""

    ps = powershell

    non_comment = [
        ln for ln in ps.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not non_comment:
        return False, "ps_has_no_logic", classification, ""

    if "```" in ps:
        return False, "markdown_fence", classification, ""

    return True, "ok", classification, ""


def audit_jsonl(jsonl_path: Path, stig_by_shortname: dict, strict_counts: bool) -> bool:
    total = 0
    failures = []
    cls_counter = Counter()
    stig_rule_counts = Counter()

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                rec = json.loads(line)
            except Exception:
                failures.append((line_no, "top_level_not_json"))
                continue

            messages = rec.get("messages") or rec.get("conversations") or []
            if not isinstance(messages, list) or len(messages) < 2:
                failures.append((line_no, "bad_messages_structure"))
                continue

            user_msg = None
            assistant_msg = None
            for m in messages:
                if m.get("role") == "user":
                    user_msg = m
                elif m.get("role") == "assistant":
                    assistant_msg = m

            if not user_msg or not assistant_msg:
                failures.append((line_no, "missing_user_or_assistant"))
                continue

            user_content = user_msg.get("content", "")
            assistant_content = assistant_msg.get("content", "")

            stig_short = infer_stig_shortname_from_user(user_content)

            ok, reason, classification, _ = validate_assistant_payload(assistant_content)
            if not ok:
                failures.append((line_no, reason))
                continue

            if classification:
                cls_counter[classification] += 1
            if stig_short:
                stig_rule_counts[stig_short] += 1

    print(f"\nFile: {jsonl_path}")
    print(f"  Total records:          {total}")
    print(f"  Classification counts:  {dict(cls_counter)}")
    print(f"  Failures:               {len(failures)}")
    if failures:
        print("  Sample failures (up to 20):")
        for ln, reason in failures[:20]:
            print(f"    line {ln}: {reason}")

    print("\nSTIG-level rule counts (from dataset):")
    for stig_short, count in stig_rule_counts.most_common():
        mapping = stig_by_shortname.get(stig_short)
        if not mapping:
            print(f"  {stig_short}: {count} rules  (NO MAPPING ENTRY)")
            continue
        mapped_total = (mapping["cat_i"] or 0) + (mapping["cat_ii"] or 0) + (mapping["cat_iii"] or 0)
        diff = count - mapped_total
        note = "  <-- MISMATCH" if strict_counts and diff != 0 else ""
        print(f"  {stig_short}: {count} rules, mapping total {mapped_total}, diff {diff}{note}")

    ok_overall = len(failures) == 0
    if strict_counts:
        for stig_short, count in stig_rule_counts.items():
            mapping = stig_by_shortname.get(stig_short)
            if not mapping:
                continue
            mapped_total = (mapping["cat_i"] or 0) + (mapping["cat_ii"] or 0) + (mapping["cat_iii"] or 0)
            if mapped_total and count != mapped_total:
                ok_overall = False

    return ok_overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping-xml", required=True, help="Path to stig_mapping.xml")
    ap.add_argument("--train-jsonl", required=True, help="Path to sft_training.jsonl")
    ap.add_argument("--val-jsonl", help="Optional path to val_integrated.jsonl")
    ap.add_argument(
        "--strict-counts",
        action="store_true",
        help="Fail if per-STIG rule count != CAT I+II+III total"
    )
    args = ap.parse_args()

    stig_by_content, stig_by_shortname = load_stig_mapping(Path(args.mapping_xml))
    print(f"Loaded {len(stig_by_shortname)} STIG entries from mapping.")

    ok_train = audit_jsonl(Path(args.train_jsonl), stig_by_shortname, args.strict_counts)

    ok_val = True
    if args.val_jsonl:
        ok_val = audit_jsonl(Path(args.val_jsonl), stig_by_shortname, args.strict_counts)

    if not (ok_train and ok_val):
        print("\nAUDIT FAILED")
        sys.exit(1)
    else:
        print("\nAUDIT PASSED")


if __name__ == "__main__":
    main()