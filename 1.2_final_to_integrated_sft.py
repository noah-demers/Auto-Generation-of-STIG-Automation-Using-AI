#!/usr/bin/env python3
import json
import argparse
import re
from pathlib import Path


SYSTEM = (
    "You are an Evaluate-STIG automation assistant.\n"
    "Given one STIG rule, you must:\n"
    "1) Classify it as AUTOMATABLE, MANUAL, or UNKNOWN.\n"
    "2) If AUTOMATABLE, output ONLY the PowerShell implementation logic that belongs\n"
    "   between the #---=== Begin Custom Code ===---# and #---=== End Custom Code ===---#\n"
    "   markers of an Evaluate-STIG module.\n"
    "   Do NOT include the Function declaration, param block, .DESCRIPTION block,\n"
    "   or the Send-CheckResult boilerplate — those are added automatically.\n"
    "   Your output is the check logic only: variable assignments, registry reads,\n"
    "   If/Else branches, and $FindingDetails construction.\n"
    "3) If MANUAL or UNKNOWN, powershell must be null.\n"
    "Output MUST be valid JSON with exactly these keys:\n"
    '{"classification":"...","powershell":...}\n'
    "Where classification is one of: AUTOMATABLE, MANUAL, UNKNOWN.\n"
    "Do not include any other text outside the JSON object."
)


MAP = {
    "automatable_confirmed": "AUTOMATABLE",
    "automatable_likely":    "AUTOMATABLE",
    "manual_likely":         "MANUAL",
    "unknown":               "UNKNOWN",
}


SIG_BLOCK_RX = re.compile(
    r"(?is)^\s*#?\s*SIG\s*#?\s*Begin\s+signature\s+block.*?^\s*#?\s*SIG\s*#?\s*End\s+signature\s+block\s*$",
    re.MULTILINE,
)


def strip_signature_blocks(ps: str) -> str:
    if not ps:
        return ps
    ps2 = re.sub(SIG_BLOCK_RX, "", ps)
    ps2 = re.sub(r"\n{4,}", "\n\n\n", ps2).strip()
    return ps2


def build_user(o: dict) -> str:
    return (
        f"STIG: {o.get('stig_shortname', '')}\n"
        f"GroupID (VulnID): {o.get('group_id', '')}\n"
        f"RuleID: {o.get('rule_id', '')}\n"
        f"Severity: {o.get('severity', '')}\n"
        f"RuleTitle: {o.get('rule_title', '')}\n\n"
        f"CheckText:\n{o.get('check_text', '')}\n\n"
        f"FixText:\n{o.get('fix_text', '')}\n\n"
        f"ProbeTypeHint: {o.get('labels', {}).get('probe_type', 'other')}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl",  required=True,
                    help="Output of build_dataset.py (final_rules_dataset.jsonl)")
    ap.add_argument("--out_jsonl", required=True,
                    help="Destination SFT file for fine-tuning")
    ap.add_argument(
        "--require_function_block",
        action="store_true",
        help="Only keep AUTOMATABLE samples where extracted=='function_block'. "
             "Recommended: filters out fallback window extractions.",
    )
    ap.add_argument(
        "--strip_signatures",
        action="store_true",
        help="Strip PowerShell code-signing signature blocks from targets.",
    )
    args = ap.parse_args()

    n = kept = skipped_no_link = skipped_no_snippet = skipped_no_block = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as f, \
         open(args.out_jsonl, "w", encoding="utf-8") as out:

        for line in f:
            n += 1
            o = json.loads(line)

            cls = MAP.get((o.get("labels") or {}).get("automatable"))
            if not cls:
                continue

            ps = None

            if cls == "AUTOMATABLE":
                if (o.get("labels") or {}).get("link_quality") != "linked":
                    skipped_no_link += 1
                    continue

                if args.require_function_block \
                        and (o.get("ps_link") or {}).get("extracted") != "function_block":
                    skipped_no_block += 1
                    continue

                ps = (o.get("ps_link") or {}).get("snippet")

                if not ps or not ps.strip():
                    skipped_no_snippet += 1
                    continue

                if args.strip_signatures:
                    ps = strip_signature_blocks(ps)

                if not ps or not ps.strip():
                    skipped_no_snippet += 1
                    continue

            target_json = json.dumps(
                {"classification": cls, "powershell": ps if cls == "AUTOMATABLE" else None},
                ensure_ascii=False,
            )

            rec = {
                "messages": [
                    {"role": "system",    "content": SYSTEM},
                    {"role": "user",      "content": build_user(o)},
                    {"role": "assistant", "content": target_json},
                ]
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Read:                 {n}")
    print(f"Wrote:                {kept}  ->  {args.out_jsonl}")
    print(f"Skipped (no link):    {skipped_no_link}")
    print(f"Skipped (no snippet): {skipped_no_snippet}")
    print(f"Skipped (no block):   {skipped_no_block}")


if __name__ == "__main__":
    main()
    