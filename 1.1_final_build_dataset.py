#!/usr/bin/env python3
import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


BEGIN_MARKER = "#---=== Begin Custom Code ===---#"
END_MARKER   = "#---=== End Custom Code ===---#"


MANUAL_PATTERNS = [
    r"\binterview\b",
    r"\breview\b.*\b(documentation|records|policy|procedure|plan|evidence)\b",
    r"\bverify\b.*\b(approved|documented|in writing|signed|training)\b",
    r"\binspect\b.*\b(records|documentation)\b",
    r"\bask\b.*\b(admin|administrator|sa|issm|isso)\b",
]


AUTO_PATTERNS = [
    r"\bregistry\b|\bhk(lm|cu|cr|u)\b",
    r"\bgpo\b|\bgroup policy\b",
    r"\bauditpol\b|\baudit policy\b",
    r"\bservice\b|\bsystemctl\b",
    r"/etc/|c:\\|/var/|/usr/|\bfile\b|\bpath\b",
    r"\bchmod\b|\bchown\b|\bacl\b|\bpermission",
    r"\brpm\b|\bdpkg\b|\bapt\b|\byum\b|\bpackage\b|\bversion\b",
    r"\bport\b|\blisten(ing)?\b|\btls\b|\bcipher\b|\bcertificate\b",
]


VULN_ID_RX = re.compile(r"\bV-(\d+)\b", re.IGNORECASE)


def classify_automatable_heuristic(text: str) -> str:
    t = (text or "").lower()
    if any(re.search(p, t) for p in MANUAL_PATTERNS):
        return "manual_likely"
    if any(re.search(p, t) for p in AUTO_PATTERNS):
        return "automatable_likely"
    return "unknown"


def infer_probe_type(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\bhk(lm|cu|cr|u)\b|\bregistry\b", t):
        return "registry"
    if re.search(r"\bgpo\b|\bgroup policy\b", t):
        return "gpo"
    if re.search(r"\bauditpol\b|\baudit policy\b", t):
        return "audit_policy"
    if re.search(r"\bservice\b|\bsystemctl\b", t):
        return "service"
    if re.search(r"/etc/|c:\\|/var/|/usr/|\bfile\b|\bpath\b", t):
        return "file"
    if re.search(r"\bchmod\b|\bchown\b|\bacl\b|\bpermission", t):
        return "permissions"
    if re.search(r"\brpm\b|\bdpkg\b|\bapt\b|\byum\b|\bpackage\b|\bversion\b", t):
        return "package"
    if re.search(r"\bport\b|\blisten(ing)?\b|\btls\b|\bcipher\b|\bcertificate\b", t):
        return "network_crypto"
    if re.search(r"\bprocess\b|\bps\b|\btasklist\b", t):
        return "process"
    return "other"


def localname(tag: str) -> str:
    return tag.split("}", 1)[-1]


def find_first_text(elem, want: str) -> str:
    for child in elem.iter():
        if localname(child.tag) == want and (child.text or "").strip():
            return child.text.strip()
    return ""


def gather_texts(elem, want: str):
    return [
        child.text.strip()
        for child in elem.iter()
        if localname(child.tag) == want and (child.text or "").strip()
    ]


def parse_xccdf_rules(xccdf_path: Path):
    tree = ET.parse(xccdf_path)
    root = tree.getroot()
    rules = []

    for group in root.iter():
        if localname(group.tag) != "Group":
            continue

        group_id = group.attrib.get("id", "") or group.attrib.get("Id", "")
        group_title = find_first_text(group, "title")

        for child in list(group):
            if localname(child.tag) != "Rule":
                continue

            check_text = ""
            for chk in child.iter():
                if localname(chk.tag) == "check-content" and (chk.text or "").strip():
                    check_text = chk.text.strip()
                    break

            rules.append({
                "group_id": group_id,
                "group_title": group_title,
                "rule_id": child.attrib.get("id", "") or child.attrib.get("Id", ""),
                "severity": child.attrib.get("severity", ""),
                "rule_title": find_first_text(child, "title"),
                "description": find_first_text(child, "description"),
                "check_text": check_text,
                "fix_text": find_first_text(child, "fixtext"),
                "refs": gather_texts(child, "ident"),
            })

    return rules


def parse_psm1_functions(ps_text: str, psm1_path: Path):
    """
    Return dict mapping VulnID ('V-243480') -> snippet (str between markers) for that function.
    """
    functions = {}

    # crude split on "Function Get-V"
    parts = re.split(r"(?im)^\s*function\s+(Get-V\d+)\s*\{", ps_text)
    # parts[0] = prelude, then (name, body, name, body, ...)
    for i in range(1, len(parts), 2):
        func_name = parts[i].strip()
        body = parts[i + 1]

        # Limit body up to matching closing brace at top level
        # Simple heuristic: find first "}\n\nFunction Get-V" or end of file
        m_next = re.search(r"(?im)^\s*\}\s*(?:#.*)?\s*^\s*Function\s+Get-V\d+\s*\{", body)
        if m_next:
            func_block = body[:m_next.start()]
        else:
            # strip final closing brace if present
            m_close = re.search(r"\}\s*$", body)
            func_block = body[:m_close.start()] if m_close else body

        # Extract .DESCRIPTION chunk to find Vuln ID
        desc_match = re.search(r"(?is)<#(.*?)#>", func_block)
        desc_text = desc_match.group(1) if desc_match else ""
        vuln_match = VULN_ID_RX.search(desc_text)
        if not vuln_match:
            continue
        vuln_id = f"V-{vuln_match.group(1)}"

        # Extract code between markers inside this function only
        b_idx = func_block.find(BEGIN_MARKER)
        e_idx = func_block.find(END_MARKER, b_idx + len(BEGIN_MARKER)) if b_idx != -1 else -1
        if b_idx == -1 or e_idx == -1 or e_idx <= b_idx:
            continue

        snippet = func_block[b_idx + len(BEGIN_MARKER):e_idx].strip()
        if not snippet:
            continue

        functions[vuln_id] = {
            "linked": True,
            "matched_id": vuln_id,
            "snippet": snippet,
            "evidence": "function_header",
            "function_name": func_name,
            "extracted": "function_block",
            "psm1_path": str(psm1_path),
        }

    return functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_jsonl", default="final_rules_dataset.jsonl")
    ap.add_argument("--out_errors", default="final_rules_errors.jsonl")
    ap.add_argument("--only_ok", action="store_true")
    args = ap.parse_args()

    n_written = 0
    n_err = 0

    with open(args.pairs_csv, newline="", encoding="utf-8") as f, \
         open(args.out_jsonl, "w", encoding="utf-8") as out, \
         open(args.out_errors, "w", encoding="utf-8") as err:

        reader = csv.DictReader(f)

        for row in reader:
            if args.only_ok and (row.get("status") or "").strip() != "OK":
                continue

            xml_p = Path(row.get("xml", ""))
            psm_p = Path(row.get("psm1", ""))

            if not xml_p.exists():
                err.write(json.dumps({
                    "type": "missing_xml",
                    "xml": str(xml_p),
                    "row": row
                }, ensure_ascii=False) + "\n")
                n_err += 1
                continue

            ps_text = psm_p.read_text(encoding="utf-8", errors="ignore") if psm_p.exists() else ""
            func_map = parse_psm1_functions(ps_text, psm_p) if ps_text else {}

            try:
                rules = parse_xccdf_rules(xml_p)
            except Exception as e:
                err.write(json.dumps({
                    "type": "parse_xccdf_failed",
                    "xml": str(xml_p),
                    "error": str(e)
                }, ensure_ascii=False) + "\n")
                n_err += 1
                continue

            for r in rules:
                combined = "\n".join([
                    r.get("rule_title", ""),
                    r.get("description", ""),
                    r.get("check_text", ""),
                    r.get("fix_text", ""),
                ])

                heuristic = classify_automatable_heuristic(combined)
                p_type = infer_probe_type(combined)

                # Map by group_id (which is V-XXXXX)
                vuln_id = r.get("group_id") or ""
                ps_link = func_map.get(vuln_id, {
                    "linked": False,
                    "matched_id": None,
                    "snippet": None,
                    "evidence": "none",
                    "function_name": None,
                    "extracted": "none",
                    "psm1_path": str(psm_p) if psm_p.exists() else None,
                })

                auto_label = "automatable_confirmed" if ps_link["extracted"] == "function_block" else heuristic

                record = {
                    "stig_shortname": row.get("ShortName", ""),
                    "stig_xccdf_path": str(xml_p),
                    "group_id": r["group_id"],
                    "group_title": r["group_title"],
                    "rule_id": r["rule_id"],
                    "rule_title": r["rule_title"],
                    "severity": r["severity"],
                    "check_text": r["check_text"],
                    "fix_text": r["fix_text"],
                    "refs": r["refs"],
                    "labels": {
                        "automatable": auto_label,
                        "automatable_heuristic": heuristic,
                        "probe_type": p_type,
                        "link_quality": "linked" if ps_link["linked"] else "unlinked",
                        "link_evidence": ps_link["evidence"],
                    },
                    "ps_link": ps_link,
                }

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"Done. Wrote {n_written} records to {args.out_jsonl}. Errors: {n_err}")


if __name__ == "__main__":
    main()