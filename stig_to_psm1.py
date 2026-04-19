#!/usr/bin/env python3
"""
stig_to_psm1.py
---------------
End-to-end: XCCDF (STIG .xml) -> Evaluate-STIG .psm1 module.


Pipeline:


    XCCDF ──► parse_xccdf_rules()            (same parser as build_dataset.py used to create the training/evaluation/testing data)
          │
          ├─► STIG-level boilerplate (module header, per-function .DESCRIPTION,
          │   param block, prelude vars, tail Send-CheckResult plumbing)
          │
          └─► for each Rule:
                build_user(rule)              (same prompt as final_to_integrated_sft.py)
                ├─► fine-tuned Qwen + LoRA    (same load path as final_eval.py)
                ├─► extract_json(response)    (same parser as final_eval.py)
                └─► classification + powershell snippet
                        │
                        ▼
                Wrap snippet between the Begin/End Custom Code markers,
                slot it into the per-function boilerplate, concatenate all
                functions under the module header, write <out>.psm1.


HuggingFace authentication
--------------------------
No token is ever hardcoded. The operator supplies it at runtime via ONE of:

    1. --hf-token <token>              (explicit CLI flag)
    2. HF_TOKEN environment variable   (export HF_TOKEN=...)
    3. HUGGING_FACE_HUB_TOKEN          (alt env var the HF libs also honor)
    4. `huggingface-cli login`         (cached to ~/.cache/huggingface/token)
    5. None                            (works only if the base model is already
                                        present in the local HF cache)

The script will NOT silently exfiltrate a token from the environment. It only
reads env vars if the caller explicitly passes --hf-token-from-env, or passes
no token at all (in which case HF's own resolver uses the cache / env as a
last resort).


Example usage:
    python stig_to_psm1.py \
        --xccdf   ./Adobe_Reader_DC_Classic_STIG.xml \
        --adapter /path/to/final_model \
        --out     ./AdobeAcrobatReaderDCClassicTrack.psm1 \
        --hf-token-from-env
"""


from __future__ import annotations


import argparse
import getpass
import hashlib
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path



# ============================================================================
# SECTION 1  —  Parsing logic copied verbatim from build_dataset.py
# ============================================================================


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
    """
    Same logic as final_build_dataset.py:parse_xccdf_rules, plus you also return
    the version / STIG ID (`<version>` inside each Rule, which Evaluate-STIG
    uses as the "STIG ID    :" line in the .DESCRIPTION block) and every
    <ident> value bucketed by system (CCI IDs go into the CCI line).
    """
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


            # You keep TWO copies of check-content:
            #   check_text     -> stripped, used for prompt + heuristics (matches training data)
            #   check_text_raw -> unmodified ElementTree text, used for CheckMD5
            #                     (Evaluate-STIG hashes the raw XCCDF text)
            check_text = ""
            check_text_raw = ""
            for chk in child.iter():
                if localname(chk.tag) == "check-content" and (chk.text or "").strip():
                    check_text_raw = chk.text
                    check_text = chk.text.strip()
                    break


            # fixtext for FixMD5 needs the same raw-vs-stripped treatment
            fix_text_raw = ""
            for ft in child.iter():
                if localname(ft.tag) == "fixtext" and (ft.text or "").strip():
                    fix_text_raw = ft.text
                    break


            # STIG ID — e.g. ARDC-CL-000005 — lives in <Rule>/<version>
            stig_id = ""
            for v in child.iter():
                if localname(v.tag) == "version" and (v.text or "").strip():
                    stig_id = v.text.strip()
                    break


            # Bucket idents by system URI so we can pull out the CCI list
            ccis = []
            for ident in child.iter():
                if localname(ident.tag) != "ident":
                    continue
                sysuri = ident.attrib.get("system", "")
                val = (ident.text or "").strip()
                if not val:
                    continue
                if "cci" in sysuri.lower():
                    ccis.append(val)


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
                # Extras needed for boilerplate only:
                "stig_id": stig_id,
                "cci_ids": ccis,
                "check_text_raw": check_text_raw,
                "fix_text_raw": fix_text_raw,
            })


    return rules



# ============================================================================
# SECTION 2  —  Prompt construction copied verbatim from final_to_integrated_sft.py
# ============================================================================


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



def build_user(o: dict) -> str:
    """Matches final_to_sft.build_user exactly. Accepts either a flat rule
    (as produced by parse_xccdf_rules + our extras) or the full SFT record."""
    probe_type = (
        (o.get("labels") or {}).get("probe_type")
        if isinstance(o.get("labels"), dict)
        else o.get("probe_type", "other")
    ) or "other"
    return (
        f"STIG: {o.get('stig_shortname', '')}\n"
        f"GroupID (VulnID): {o.get('group_id', '')}\n"
        f"RuleID: {o.get('rule_id', '')}\n"
        f"Severity: {o.get('severity', '')}\n"
        f"RuleTitle: {o.get('rule_title', '')}\n\n"
        f"CheckText:\n{o.get('check_text', '')}\n\n"
        f"FixText:\n{o.get('fix_text', '')}\n\n"
        f"ProbeTypeHint: {probe_type}\n"
    )



# ============================================================================
# SECTION 3  —  JSON extractor copied verbatim from final_eval.py
# ============================================================================


def extract_json(text: str):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    return text[start:end] if end != -1 else None



# ============================================================================
# SECTION 4  —  STIG-level metadata from the XCCDF root
# ============================================================================


def extract_stig_meta(xccdf_path: Path) -> dict:
    """
    Pull the fields Evaluate-STIG puts in the module banner:
        STIG     -> benchmark title (without the trailing 'Security Technical
                    Implementation Guide' suffix)
        Version  -> 'V{version}R{release}' assembled from <version> and
                    plain-text release-info
        ShortName-> benchmark id with common filler stripped, good for the
                    output .psm1 filename / the SFT `stig_shortname` field
    """
    tree = ET.parse(xccdf_path)
    root = tree.getroot()


    bench_id = root.attrib.get("id", "") or ""


    title = ""
    version = ""
    release = ""


    # <title> at the benchmark level (first direct child title)
    for child in root:
        if localname(child.tag) == "title" and (child.text or "").strip():
            title = child.text.strip()
            break


    # <version>…</version> — benchmark version = the 'V' number
    for child in root:
        if localname(child.tag) == "version" and (child.text or "").strip():
            version = child.text.strip()
            break


    # release comes from <plain-text id="release-info">Release: N Benchmark Date…</plain-text>
    for child in root:
        if localname(child.tag) != "plain-text":
            continue
        if child.attrib.get("id", "") == "release-info" and (child.text or "").strip():
            m = re.search(r"Release:\s*([0-9.]+)", child.text)
            if m:
                release = m.group(1)
            break


    # Make a clean STIG shortname (strip the "Security Technical
    # Implementation Guide" / "STIG" suffix).
    short_title = title
    for suffix in (
        " Security Technical Implementation Guide",
        " STIG",
    ):
        if short_title.endswith(suffix):
            short_title = short_title[: -len(suffix)]


    # Filename-safe shortname (no spaces)
    filename_short = re.sub(r"[^A-Za-z0-9]+", "", short_title) or \
                     re.sub(r"[^A-Za-z0-9]+", "", bench_id) or "Module"


    version_str = ""
    if version and release:
        version_str = f"V{version}R{release}"
    elif version:
        version_str = f"V{version}R1"


    return {
        "bench_id": bench_id,
        "title_full": title,
        "title_short": short_title.strip() or bench_id,
        "filename_short": filename_short,
        "version_str": version_str or "V1R1",
    }



# ============================================================================
# SECTION 5  —  PSM1 boilerplate assembly
# ============================================================================


MODULE_HEADER_TMPL = '''\
##########################################################################
# Evaluate-STIG module
# --------------------
# STIG:     {title_short}
# Version:  {version_str}
# Class:    UNCLASSIFIED
# Updated:  {updated}
##########################################################################
$ErrorActionPreference = "Stop"
'''



def _md5_hex(s: str) -> str:
    return hashlib.md5((s or "").encode("utf-8")).hexdigest().upper()



def _discuss_text(description_field: str) -> str:
    """VulnDiscussion is embedded as escaped XML inside <description>. Pull
    the inner body so we can hash it like Evaluate-STIG does for DiscussMD5."""
    if not description_field:
        return ""
    unescaped = html.unescape(description_field)
    m = re.search(r"<VulnDiscussion>(.*?)</VulnDiscussion>", unescaped, re.DOTALL)
    return m.group(1) if m else unescaped



def _function_description(rule: dict) -> str:
    """The .DESCRIPTION docstring block, matching the ground-truth format."""
    vuln_id = rule.get("group_id") or ""
    stig_id = rule.get("stig_id") or ""
    rule_id = rule.get("rule_id") or ""
    # Evaluate-STIG sorts CCI IDs numerically (by the digits after "CCI-"),
    # not by their order in the XCCDF.
    _raw_ccis = rule.get("cci_ids") or []
    def _cci_sort_key(c):
        m = re.search(r"(\d+)", c or "")
        return int(m.group(1)) if m else 0
    cci_ids = ", ".join(sorted(_raw_ccis, key=_cci_sort_key)) or ""
    rule_name = rule.get("group_title") or ""
    # Rule title cleanup: drop CR/LF, collapse runs of whitespace
    rule_title = (rule.get("rule_title") or "").replace("\r", "").replace("\n", " ")
    rule_title = rule_title.replace("\u00a0", " ")
    rule_title = re.sub(r"[ \t]+", " ", rule_title).strip()
    # Evaluate-STIG hashes the UNSTRIPPED XCCDF text for CheckMD5 / FixMD5.
    check_src = rule.get("check_text_raw")
    if check_src is None or check_src == "":
        check_src = rule.get("check_text") or ""
    fix_src = rule.get("fix_text_raw")
    if fix_src is None or fix_src == "":
        fix_src = rule.get("fix_text") or ""


    discuss_md5 = _md5_hex(_discuss_text(rule.get("description") or ""))
    check_md5   = _md5_hex(check_src)
    fix_md5     = _md5_hex(fix_src)


    return (
        "    <#\n"
        "    .DESCRIPTION\n"
        f"        Vuln ID    : {vuln_id}\n"
        f"        STIG ID    : {stig_id}\n"
        f"        Rule ID    : {rule_id}\n"
        f"        CCI ID     : {cci_ids}\n"
        f"        Rule Name  : {rule_name}\n"
        f"        Rule Title : {rule_title}\n"
        f"        DiscussMD5 : {discuss_md5}\n"
        f"        CheckMD5   : {check_md5}\n"
        f"        FixMD5     : {fix_md5}\n"
        "    #>\n"
    )



# Param block + prelude variables every Evaluate-STIG function shares.
FUNCTION_PARAM_AND_PRELUDE = '''\
    param (
        [Parameter(Mandatory = $true)]
        [String]$ScanType,

        [Parameter(Mandatory = $false)]
        [String]$AnswerFile,

        [Parameter(Mandatory = $false)]
        [String]$AnswerKey,

        [Parameter(Mandatory = $false)]
        [String]$Instance,

        [Parameter(Mandatory = $false)]
        [String]$Database,

        [Parameter(Mandatory = $false)]
        [String]$SiteName
    )

    $ModuleName = (Get-Command $MyInvocation.MyCommand).Source
    $FuncDescription = ($MyInvocation.MyCommand.ScriptBlock -split "#>")[0].split("`r`n")
    $VulnID = ($FuncDescription | Select-String -Pattern "V-\\d{4,6}$").Matches[0].Value
    $RuleID = ($FuncDescription | Select-String -Pattern "SV-\\d{4,6}r\\d{1,}_rule$").Matches[0].Value
    $Status = "Not_Reviewed"  # Acceptable values are 'Not_Reviewed', 'Open', 'NotAFinding', 'Not_Applicable'
    $FindingDetails = ""
    $Comments = ""
    $AFStatus = ""
    $SeverityOverride = ""  # Acceptable values are 'CAT_I', 'CAT_II', 'CAT_III'.  Only use if STIG calls for a severity change based on specified critera.
    $Justification = ""  # If SeverityOverride is used, a justification is required.
    # $ResultObject = [System.Collections.Generic.List[System.Object]]::new()
'''



# Tail — everything after the End Custom Code marker up through the closing `}`.
FUNCTION_TAIL = '''\

    if ($FindingDetails.Trim().Length -gt 0) {
        $ResultHash = Get-TextHash -Text $FindingDetails -Algorithm SHA1
    }
    else {
        $ResultHash = ""
    }

    if ($PSBoundParameters.AnswerFile) {
        $GetCorpParams = @{
            AnswerFile   = $PSBoundParameters.AnswerFile
            VulnID       = $VulnID
            RuleID       = $RuleID
            AnswerKey    = $PSBoundParameters.AnswerKey
            Status       = $Status
            Hostname     = $Hostname
            Username     = $Username
            UserSID      = $UserSID
            Instance     = $Instance
            Database     = $Database
            Site         = $Site
            ResultHash   = $ResultHash
            ResultData   = $FindingDetails
            ShowRun      = $ShowRunningConfig
            ESPath       = $ESPath
            LogPath      = $LogPath
            LogComponent = $LogComponent
            OSPlatform   = $OSPlatform
        }

        $AnswerData = (Get-CorporateComment @GetCorpParams)
        if ($Status -eq $AnswerData.ExpectedStatus) {
            $AFKey = $AnswerData.AFKey
            $AFStatus = $AnswerData.AFStatus
            $Comments = $AnswerData.AFComment | Out-String
        }
    }

    $SendCheckParams = @{
        Module           = $ModuleName
        Status           = $Status
        FindingDetails   = $FindingDetails
        AFKey            = $AFkey
        AFStatus         = $AFStatus
        Comments         = $Comments
        SeverityOverride = $SeverityOverride
        Justification    = $Justification
        HeadInstance     = $Instance
        HeadDatabase     = $Database
        HeadSite         = $Site
        HeadHash         = $ResultHash
    }
    if ($AF_UserHeader) {
        $SendCheckParams.Add("HeadUsername", $Username)
        $SendCheckParams.Add("HeadUserSID", $UserSID)
    }

    return Send-CheckResult @SendCheckParams
}
'''



MANUAL_BODY = (
    '    $Status = "Not_Reviewed"\n'
    '    $FindingDetails = "This check must be performed manually." | Out-String\n'
)


UNKNOWN_BODY = (
    '    $Status = "Not_Reviewed"\n'
    '    $FindingDetails = "Automation status undetermined; perform manual review." | Out-String\n'
)



def _indent_snippet(ps: str, indent: str = "    ") -> str:
    """Indent model output to 4 spaces. If the model already indented, leave
    its relative structure alone — just make sure no line is less-indented
    than the 4-space function scope."""
    if not ps:
        return ""
    lines = ps.splitlines()
    # Strip trailing blank lines
    while lines and not lines[-1].strip():
        lines.pop()
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        # If the line already starts with >=4 spaces, keep as-is.
        if ln.startswith("    ") or ln.startswith("\t"):
            out.append(ln)
        else:
            out.append(indent + ln)
    return "\n".join(out) + "\n"



def build_function(rule: dict, classification: str, powershell: str | None) -> str:
    """Assemble one `Function Get-V##### { ... }` block. Returns an empty
    string for rules without a V- id (Evaluate-STIG can't route them)."""
    vuln_id = rule.get("group_id") or ""
    m = VULN_ID_RX.search(vuln_id)
    if not m:
        return ""


    func_name = f"Get-V{m.group(1)}"


    desc_block    = _function_description(rule)
    param_prelude = FUNCTION_PARAM_AND_PRELUDE


    if classification == "AUTOMATABLE" and powershell and powershell.strip():
        body = _indent_snippet(powershell.strip())
    elif classification == "MANUAL":
        body = MANUAL_BODY
    else:
        body = UNKNOWN_BODY


    begin_end_block = (
        f"\n    {BEGIN_MARKER}\n"
        f"{body}"
        f"    {END_MARKER}\n"
    )


    return (
        f"Function {func_name} {{\n"
        f"{desc_block}\n"
        f"{param_prelude}"
        f"{begin_end_block}"
        f"{FUNCTION_TAIL}"
    )



# ============================================================================
# SECTION 6  —  HuggingFace token resolution (no hardcoding anywhere)
# ============================================================================


def resolve_hf_token(cli_token: str | None,
                     from_env: bool,
                     prompt_if_missing: bool) -> str | None:
    """
    Returns the HF token to hand to `from_pretrained`, or None if none is
    available (in which case we rely on the HF library's own resolver, i.e.
    the local cache or `huggingface-cli login` state).

    Precedence, top wins:
        1. --hf-token <value> on the command line
        2. --hf-token-from-env       -> HF_TOKEN, HUGGING_FACE_HUB_TOKEN
        3. --hf-token-prompt         -> interactive getpass() prompt
        4. None                      -> let HF resolve from cache / login

    This function NEVER reads the environment unless --hf-token-from-env
    was passed. That avoids surprise-exfil in shared / CI environments.
    """
    if cli_token:
        return cli_token

    if from_env:
        for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            val = os.environ.get(var)
            if val:
                return val
        print(f"[warn] --hf-token-from-env was set but no HF_TOKEN / "
              f"HUGGING_FACE_HUB_TOKEN was found in the environment.",
              file=sys.stderr)

    if prompt_if_missing:
        # getpass masks the input so it never appears in terminal scrollback
        try:
            return getpass.getpass("HuggingFace token (hidden): ").strip() or None
        except (EOFError, KeyboardInterrupt):
            print("\n[abort] No token provided.", file=sys.stderr)
            sys.exit(2)

    return None



# ============================================================================
# SECTION 7  —  Inference (fine-tuned Qwen + LoRA). Heavy ML imports happen
#               inside the class so that --help / argparse errors don't pay
#               the cost of loading torch.
# ============================================================================


# Upper bound on generated tokens per rule. The fine-tuned model stops on its
# own eos_token_id; this is just a safety ceiling that mirrors final_eval.py.
_MAX_NEW_TOKENS = 4096


class FineTunedRunner:
    """Loads the fine-tuned adapter once, generates per rule.

    The base model identity is read from the adapter's own
    ``adapter_config.json`` (the ``base_model_name_or_path`` field that PEFT
    records at training time), so the caller only has to supply the adapter
    directory — there's no chance of pairing the adapter with the wrong base.
    """


    def __init__(self, adapter_dir: str, hf_token: str | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel, PeftConfig


        self.torch = torch


        # Pull the base model ID straight from the adapter that was produced
        # by the training run. This is the single source of truth.
        peft_cfg = PeftConfig.from_pretrained(adapter_dir)
        base_model = peft_cfg.base_model_name_or_path
        if not base_model:
            sys.exit(f"[abort] Adapter at {adapter_dir} is missing "
                     f"'base_model_name_or_path' in adapter_config.json.")
        print(f"[infer] Adapter targets base model: {base_model}", flush=True)


        # `token=None` is the documented signal to let HF resolve via cache
        # or `huggingface-cli login` — so passing None here is safe and means
        # "use whatever auth is already wired up on this machine".
        tok = AutoTokenizer.from_pretrained(
            base_model, use_fast=True, token=hf_token, trust_remote_code=True
        )


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )


        num_gpus = max(1, min(2, torch.cuda.device_count())) if torch.cuda.is_available() else 0
        max_memory = {i: "40GiB" for i in range(num_gpus)} if num_gpus else None
        if max_memory is not None:
            max_memory["cpu"] = "48GiB"


        print(f"[infer] Loading base model {base_model}…", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            token=hf_token,
            trust_remote_code=True,
        )


        print(f"[infer] Loading LoRA adapter {adapter_dir}…", flush=True)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()


        self.tok = tok
        self.model = model
        self.first_device = next(model.parameters()).device


    def generate(self, user_msg: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tok(prompt, return_tensors="pt").to(self.first_device)
        prompt_len = inputs["input_ids"].shape[1]


        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()



def classify_and_generate(runner: FineTunedRunner,
                          rule_for_prompt: dict) -> tuple[str, str | None]:
    """Returns (classification, powershell_or_None) from one model call."""
    raw = runner.generate(build_user(rule_for_prompt))
    jtxt = extract_json(raw)
    if not jtxt:
        print(f"[warn] {rule_for_prompt.get('group_id')}: no JSON in response", flush=True)
        return "UNKNOWN", None


    # Same escape-fix as final_eval.py
    jtxt = jtxt.replace('\\`', '`')


    try:
        obj = json.loads(jtxt)
    except Exception as e:
        print(f"[warn] {rule_for_prompt.get('group_id')}: JSON parse error: {e}", flush=True)
        return "UNKNOWN", None


    cls = (obj.get("classification") or "UNKNOWN").upper()
    if cls not in ("AUTOMATABLE", "MANUAL", "UNKNOWN"):
        cls = "UNKNOWN"
    ps = obj.get("powershell")
    if cls != "AUTOMATABLE":
        ps = None
    return cls, ps



# ============================================================================
# SECTION 8  —  main()
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Generate an Evaluate-STIG .psm1 from an XCCDF STIG file "
                    "using a fine-tuned Qwen + LoRA model."
    )
    ap.add_argument("--xccdf", required=True, type=Path,
                    help="Path to the XCCDF .xml STIG")
    ap.add_argument("--out", required=True, type=Path,
                    help="Destination .psm1 path")
    ap.add_argument("--adapter", required=True, type=str,
                    help="Path to the LoRA adapter (final_model/). The base "
                         "model ID is read from adapter_config.json.")

    # ---- HF token options: MUTUALLY EXCLUSIVE, NO DEFAULTS, NO HARDCODING ----
    tok_group = ap.add_mutually_exclusive_group()
    tok_group.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace access token (explicit). Consider --hf-token-from-env "
             "or --hf-token-prompt to avoid leaving the token in shell history.",
    )
    tok_group.add_argument(
        "--hf-token-from-env", action="store_true",
        help="Read the token from the HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
             "environment variable.",
    )
    tok_group.add_argument(
        "--hf-token-prompt", action="store_true",
        help="Prompt for the token interactively (hidden input).",
    )

    args = ap.parse_args()


    if not args.xccdf.exists():
        sys.exit(f"XCCDF not found: {args.xccdf}")


    # Resolve the HF token (may legitimately be None if the base model is
    # already present in the local HF cache or `huggingface-cli login`
    # has been run on this host).
    hf_token = resolve_hf_token(
        cli_token=args.hf_token,
        from_env=args.hf_token_from_env,
        prompt_if_missing=args.hf_token_prompt,
    )


    # 1. Parse the STIG
    meta  = extract_stig_meta(args.xccdf)
    rules = parse_xccdf_rules(args.xccdf)


    print(f"[stig] {meta['title_short']}  {meta['version_str']}", flush=True)
    print(f"[stig] {len(rules)} rule(s) parsed from {args.xccdf}", flush=True)


    # 2. Load the fine-tuned model. The base model ID is read from the
    #    adapter's own adapter_config.json (recorded at training time), so
    #    the operator only has to point us at --adapter.
    runner = FineTunedRunner(
        adapter_dir=args.adapter,
        hf_token=hf_token,
    )


    # 3. Emit the module header
    chunks = [MODULE_HEADER_TMPL.format(
        title_short=meta["title_short"],
        version_str=meta["version_str"],
        updated=datetime.now().strftime("%m/%d/%Y"),
    )]


    # 4. Per-rule: build prompt, run model, wrap body in boilerplate
    stats = {"AUTOMATABLE": 0, "MANUAL": 0, "UNKNOWN": 0, "skipped": 0}


    for i, rule in enumerate(rules, 1):
        vid = rule.get("group_id") or ""
        if not VULN_ID_RX.search(vid):
            stats["skipped"] += 1
            continue


        # Match build_user()'s expected shape + add stig_shortname
        prompt_obj = dict(rule)
        prompt_obj["stig_shortname"] = meta["title_short"]
        prompt_obj["probe_type"] = infer_probe_type("\n".join([
            rule.get("rule_title", ""),
            rule.get("check_text", ""),
            rule.get("fix_text", ""),
        ]))


        print(f"[{i}/{len(rules)}] {vid}  …", flush=True)
        cls, ps = classify_and_generate(runner, prompt_obj)
        stats[cls] += 1


        fn_text = build_function(rule, cls, ps)
        if fn_text:
            chunks.append(fn_text)


    # 5. Concatenate + write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chunks), encoding="utf-8")


    print("\n=== summary ===", flush=True)
    for k, v in stats.items():
        print(f"  {k:12} : {v}")
    print(f"Wrote: {args.out.resolve()}", flush=True)



if __name__ == "__main__":
    main()
