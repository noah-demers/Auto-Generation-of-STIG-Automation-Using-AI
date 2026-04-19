#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/mnt/lustre/koa/scratch/demersn/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/lustre/koa/scratch/demersn/hf_cache"

import json
import re
import difflib
import argparse
from collections import Counter
from typing import Optional, Dict, Any, List, Tuple, Iterable, Set

import ctypes
from tree_sitter import Language, Parser

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
HF_TOKEN = "THE_HUGGING_FACE_TOKEN_HERE"

SIMILARITY_THRESHOLD = 0.70
STRING_OVERLAP_THRESHOLD = 0.90
STRUCTURE_SIMILARITY_THRESHOLD = 0.55
CRYSTAL_SIMILARITY_THRESHOLD = 0.65
TRIVIAL_NGRAM_PERCENTILE = 0.90
MAX_TRIVIAL_NGRAMS = 5000

RESERVED_VARS = {
    "$status", "$findingdetails", "$vulnid", "$ruleid", "$modulename",
    "$comments", "$afstatus", "$severityoverride", "$justification",
    "$resulthash", "$showrunningconfig", "$hostname", "$username",
    "$usersid", "$instance", "$database", "$sitename", "$espath",
    "$logpath", "$logcomponent", "$osplatform",
}

KEYWORD_PATTERN = re.compile(
    r"\b(if|elseif|else|foreach|for|while|switch|try|catch|finally|return|function|param)\b",
    flags=re.IGNORECASE,
)

OPERATOR_PATTERN = re.compile(
    r"-(eq|ne|gt|ge|lt|le|like|notlike|match|notmatch|contains|notcontains|in|notin|and|or|not|is|isnot|replace)",
    flags=re.IGNORECASE,
)

HELPER_PATTERN = re.compile(
    r"\b(get-registryresult|get-ad\w+|get-membersofadgroup|get-groupmembership|get-itemproperty|get-itempropertyvalue|test-path|select-object|out-string)\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Real PowerShell AST via tree-sitter (Airbus grammar)
# ---------------------------------------------------------------------------

def _init_powershell_parser() -> Parser:
    lib = os.environ.get(
        "TS_POWERSHELL_LIB",
        "/home/demersn/ts-langs/airbus-tree-sitter-powershell/parser.so",
    )
    powershell_lib = ctypes.cdll.LoadLibrary(lib)
    func = powershell_lib.tree_sitter_powershell
    func.restype = ctypes.c_void_p
    handle = func()
    lang = Language(handle)
    parser = Parser(lang)
    return parser


PS_PARSER = _init_powershell_parser()

_TERMINAL_MAP = {
    "variable": "VAR",
    "integer_literal": "NUM",
    "decimal_integer_literal": "NUM",
    "string_literal": "STR",
    "expandable_string_literal": "STR",
    "comparison_operator": "OP",
    "assignment_operator": "ASSIGN",
}

_STATEMENT_HEADS = {
    "if_statement": "IF",
    "while_statement": "WHILE",
    "for_statement": "FOR",
    "foreach_statement": "FOREACH",
    "function_statement": "FUNC",
    "switch_statement": "SWITCH",
    "try_statement": "TRY",
    "catch_clause": "CATCH",
    "finally_clause": "FINALLY",
}


def _walk_ast(node, out: List[str]) -> None:
    t = node.type

    if t in _STATEMENT_HEADS:
        out.append(_STATEMENT_HEADS[t])

    if t in _TERMINAL_MAP:
        out.append(_TERMINAL_MAP[t])

    if t == "statement_block":
        out.append("LBRACE")
    elif t == "param_block":
        out.append("LPARAM")

    if t in ("pipe", "pipeline", "pipe_operator"):
        out.append("PIPE")

    for child in node.children:
        _walk_ast(child, out)

    if t == "statement_block":
        out.append("RBRACE")
    elif t == "param_block":
        out.append("RPARAM")


def powershell_structure_sequence(text: Optional[str]) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        tree = PS_PARSER.parse(text.encode("utf-8"))
    except Exception:
        return ""
    root = tree.root_node
    if root.has_error:
        pass
    tokens: List[str] = []
    _walk_ast(root, tokens)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Normalization and tokenization
# ---------------------------------------------------------------------------

# Covers $env:windir, $global:status, ${env:windir}
_VAR_PATTERN = re.compile(r"\$(?:\w+(?::\w+)?|\{[^}]+\})")


def semantic_normalize_ps(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""

    s = re.sub(r"<#.*?#>", "", text, flags=re.DOTALL)
    s = re.sub(r"#.*", "", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)

    var_map: Dict[str, str] = {}
    counter = 0
    for var in _VAR_PATTERN.findall(s):
        if var not in RESERVED_VARS and var not in var_map:
            var_map[var] = f"$v{counter}"
            counter += 1

    return _VAR_PATTERN.sub(lambda m: var_map.get(m.group(0), m.group(0)), s)


def tokenize_for_crystal(text: Optional[str]) -> List[str]:
    s = semantic_normalize_ps(text)
    if not s:
        return []

    s = re.sub(
        r"(['\"])(.*?)\1",
        lambda m: " " + re.sub(r"\s+", "_", m.group(2).strip()) + " ",
        s,
    )
    s = re.sub(r"\s+", " ", s).strip()

    # FIX 1: variable token uses same robust pattern as _VAR_PATTERN —
    # scoped vars like $env:windir are never split at the colon
    return re.findall(
        r"\$(?:\w+(?::\w+)?|\{[^}]+\})|[a-zA-Z_][a-zA-Z0-9_./\\:-]*|-[a-z]+|\d+|\{\{|\}|\(|\)|\[|\]|\+=|==|=|\.\w+|[^\s]",
        s,
    )


def ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])


# ---------------------------------------------------------------------------
# CrystalBLEU-style trivial n-gram filtering
# ---------------------------------------------------------------------------

def build_trivial_ngram_set(dataset, split_name: str = "train") -> Set[Tuple[str, ...]]:
    counts: Counter = Counter()

    for ex in dataset:
        messages = ex["messages"]
        try:
            gt_obj = json.loads(messages[2]["content"])
        except Exception:
            continue

        gt_ps = gt_obj.get("powershell")
        if not isinstance(gt_ps, str) or not gt_ps.strip():
            continue

        toks = tokenize_for_crystal(gt_ps)
        counts.update(set(ngrams(toks, 2)))
        counts.update(set(ngrams(toks, 3)))

    if not counts:
        return set()

    freq_values = sorted(counts.values())
    cutoff_index = max(0, min(len(freq_values) - 1, int(len(freq_values) * TRIVIAL_NGRAM_PERCENTILE) - 1))
    cutoff = freq_values[cutoff_index]

    # ranked is a list of (ngram, count) tuples sorted by descending frequency —
    # it has a guaranteed stable order
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    # FIX 2+4: slice the ordered list directly before discarding counts —
    # deterministic top-N most frequent n-grams, cutoff enforced in one pass,
    # no intermediate unordered set, no potential NameError
    trivial = {item[0] for item in ranked[:MAX_TRIVIAL_NGRAMS] if item[1] >= cutoff}

    print(
        f"[{split_name}] Built {len(trivial)} trivial n-grams "
        f"from {len(counts)} total n-grams (cutoff freq={cutoff}).",
        flush=True,
    )
    return trivial


def remove_trivial_ngrams(tokens: List[str], trivial_ngrams: Set[Tuple[str, ...]]) -> List[str]:
    if not tokens or not trivial_ngrams:
        return tokens[:]

    keep = [True] * len(tokens)

    for n in (3, 2):
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i:i+n])
            if ng in trivial_ngrams:
                for j in range(i, i + n):
                    keep[j] = False

    filtered = [tok for tok, k in zip(tokens, keep) if k]
    return filtered if filtered else tokens[:]


def crystal_weighted_sequence(text: Optional[str], trivial_ngrams: Set[Tuple[str, ...]]) -> str:
    toks = tokenize_for_crystal(text)
    toks = remove_trivial_ngrams(toks, trivial_ngrams)
    return " ".join(toks)


def crystal_similarity(pred_ps: str, gt_ps: str, trivial_ngrams: Set[Tuple[str, ...]]) -> float:
    pred_seq = crystal_weighted_sequence(pred_ps, trivial_ngrams)
    gt_seq = crystal_weighted_sequence(gt_ps, trivial_ngrams)

    if not pred_seq or not gt_seq:
        return 0.0

    return difflib.SequenceMatcher(None, pred_seq, gt_seq).ratio()


# ---------------------------------------------------------------------------
# AST-backed structural comparison
# ---------------------------------------------------------------------------

def structural_similarity(pred_ps: str, gt_ps: str) -> float:
    pred_seq = powershell_structure_sequence(pred_ps)
    gt_seq = powershell_structure_sequence(gt_ps)

    if not pred_seq or not gt_seq:
        return 0.0

    return difflib.SequenceMatcher(None, pred_seq, gt_seq).ratio()


# ---------------------------------------------------------------------------
# Additional semantic signal
# ---------------------------------------------------------------------------

def essential_string_overlap(pred_ps: str, gt_ps: str) -> float:
    gt_strings = {s.lower() for s in re.findall(r"['\"]([^'\"]{4,})['\"]", gt_ps or "")}
    pred_strings = {s.lower() for s in re.findall(r"['\"]([^'\"]{4,})['\"]", pred_ps or "")}

    if not gt_strings:
        return 0.0

    return len(gt_strings & pred_strings) / len(gt_strings)


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------

def score_powershell(
    pred_ps: str,
    gt_ps: str,
    trivial_ngrams: Set[Tuple[str, ...]],
) -> Tuple[bool, str, float, Dict[str, float]]:
    norm_pred = semantic_normalize_ps(pred_ps)
    norm_gt = semantic_normalize_ps(gt_ps)

    lexical_ratio = difflib.SequenceMatcher(None, norm_pred, norm_gt).ratio()
    struct_ratio = structural_similarity(pred_ps, gt_ps)
    crystal_ratio = crystal_similarity(pred_ps, gt_ps, trivial_ngrams)
    str_overlap = essential_string_overlap(pred_ps, gt_ps)

    diagnostics = {
        "lexical_ratio": round(lexical_ratio, 4),
        "structural_ratio": round(struct_ratio, 4),
        "crystal_ratio": round(crystal_ratio, 4),
        "string_overlap": round(str_overlap, 4),
    }

    if crystal_ratio >= CRYSTAL_SIMILARITY_THRESHOLD:
        return True, "high_crystal_match", crystal_ratio, diagnostics

    # FIX 3: use STRUCTURE_SIMILARITY_THRESHOLD constant — no more hardcoded 0.60
    if lexical_ratio >= SIMILARITY_THRESHOLD and struct_ratio >= STRUCTURE_SIMILARITY_THRESHOLD:
        return True, "high_similarity", lexical_ratio, diagnostics

    # slight buffer below the main threshold for string+structure path
    if str_overlap >= STRING_OVERLAP_THRESHOLD and struct_ratio >= STRUCTURE_SIMILARITY_THRESHOLD - 0.05:
        return True, "string_plus_structure", str_overlap, diagnostics

    return False, "mismatch", max(lexical_ratio, crystal_ratio, struct_ratio), diagnostics


# ---------------------------------------------------------------------------
# JSON extraction and checkpointing
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Optional[str]:
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


def load_checkpoint(path: str) -> Optional[Dict]:
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if "seen_indices" in data:
                return data
        except Exception:
            pass
    return None


def save_checkpoint(path: str, seen_indices: set, m: dict, f_logs: list, trivial_ngrams_count: int):
    with open(path, "w") as f:
        json.dump(
            {
                "seen_indices": list(seen_indices),
                "summary_metrics": m,
                "trivial_ngrams_count": trivial_ngrams_count,
                "detailed_failure_audit": f_logs,
            },
            f,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def eval_model(adapter_dir: str, test_jsonl: str, results_dir: str, train_jsonl: Optional[str] = None):
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "final_results_report.json")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, token=HF_TOKEN)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    num_gpus = min(2, torch.cuda.device_count())
    max_memory = {i: "40GiB" for i in range(num_gpus)}
    max_memory["cpu"] = "48GiB"
    print(f"Using {num_gpus} GPU(s). Max memory per GPU: 40 GiB", flush=True)

    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        token=HF_TOKEN,
    )

    print("Loading LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    first_device = next(model.parameters()).device

    ds = load_dataset("json", data_files={"test": test_jsonl})["test"]
    print(f"Test set size: {len(ds)}", flush=True)

    trivial_ngrams: Set[Tuple[str, ...]] = set()
    if train_jsonl:
        train_ds = load_dataset("json", data_files={"train": train_jsonl})["train"]
        print(f"Train set size: {len(train_ds)}", flush=True)
        trivial_ngrams = build_trivial_ngram_set(train_ds, split_name="train")
    else:
        print("[WARN] No --train-jsonl provided; CrystalBLEU-style weighting disabled.", flush=True)

    checkpoint = load_checkpoint(report_path)
    if checkpoint:
        seen_indices: set = set(checkpoint["seen_indices"])
        m: Dict[str, Any] = checkpoint["summary_metrics"]
        f_logs: List[Dict] = checkpoint["detailed_failure_audit"]
        print(f"[RESUME] Skipping {len(seen_indices)} already-processed items.", flush=True)
    else:
        seen_indices = set()
        f_logs = []
        m = {
            "total_processed": 0,
            "correct_cls": 0,
            "gt_auto_count": 0,
            "pred_auto_on_gt": 0,
            "func_correct": 0,
            "fully_correct": 0,
            "fail_cls": 0,
            "fail_coverage": 0,
            "fail_func": 0,
        }

    for idx, ex in enumerate(ds):
        if idx in seen_indices:
            continue

        messages = ex["messages"]
        gt_obj = json.loads(messages[2]["content"])
        gt_cls = gt_obj["classification"]
        gt_ps = gt_obj.get("powershell")

        prompt = tok.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(first_device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
            )

        response = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        jtxt = extract_json(response)

        if jtxt:
            jtxt = jtxt.replace('\\`', '`')

        m["total_processed"] += 1

        # All per-item variables initialized unconditionally before any branch
        cls_ok = False
        ok = False
        method = "n/a"
        score = 0.0
        diagnostics: Dict[str, float] = {}
        current_fails: List[str] = []
        pred_cls = pred_ps = None

        if jtxt:
            try:
                obj = json.loads(jtxt)
                pred_cls = obj.get("classification")
                pred_ps = obj.get("powershell")

                if pred_cls == gt_cls:
                    m["correct_cls"] += 1
                    cls_ok = True
                else:
                    m["fail_cls"] += 1
                    current_fails.append(
                        f"CLASS_MISMATCH: got={pred_cls} expected={gt_cls}"
                    )

                if gt_cls == "AUTOMATABLE":
                    m["gt_auto_count"] += 1

                    if (
                        pred_cls == "AUTOMATABLE"
                        and isinstance(pred_ps, str)
                        and pred_ps.strip()
                    ):
                        m["pred_auto_on_gt"] += 1
                        ok, method, score, diagnostics = score_powershell(
                            pred_ps, gt_ps or "", trivial_ngrams
                        )
                        if ok:
                            m["func_correct"] += 1
                        else:
                            m["fail_func"] += 1
                            current_fails.append(
                                f"FUNCTIONAL_LOGIC_FAIL "
                                f"(method={method}, score={score:.3f}, diagnostics={diagnostics})"
                            )
                    else:
                        ok = False
                        m["fail_coverage"] += 1
                        current_fails.append("COVERAGE_FAIL")
                else:
                    # MANUAL/UNKNOWN: classification is the only gate
                    ok = True

            except Exception as e:
                current_fails.append(f"JSON_PARSE_ERROR: {e}")
        else:
            current_fails.append("NO_JSON_FOUND")

        if gt_cls == "AUTOMATABLE":
            if cls_ok and ok:
                m["fully_correct"] += 1
        else:
            if cls_ok:
                m["fully_correct"] += 1

        if current_fails:
            entry: Dict[str, Any] = {
                "idx": idx,
                "failure_reasons": current_fails,
                "gt_classification": gt_cls,
                "pred_classification": pred_cls,
                "input_prompt": prompt,
                "ai_prediction_raw": response,
            }
            if any("FUNCTIONAL_LOGIC_FAIL" in r for r in current_fails):
                entry["similarity_score"] = round(score, 4)
                entry["similarity_method"] = method
                entry["diagnostics"] = diagnostics
                entry["predicted_powershell"] = pred_ps
                entry["gold_powershell"] = gt_ps
            f_logs.append(entry)

        seen_indices.add(idx)

        if len(seen_indices) % 10 == 0:
            print(
                f"[{len(seen_indices)}/{len(ds)}] "
                f"Fully Correct: {m['fully_correct']} | "
                f"Cls Acc: {m['correct_cls']}/{m['total_processed']} | "
                f"Func Acc: {m['func_correct']}/{max(m['pred_auto_on_gt'], 1)}",
                flush=True,
            )

        if len(seen_indices) % 5 == 0:
            save_checkpoint(report_path, seen_indices, m, f_logs, len(trivial_ngrams))

    total = m["total_processed"]
    unique_fails = len(f_logs)

    cls_acc = (m["correct_cls"] / total) * 100 if total else 0.0
    cov = (m["pred_auto_on_gt"] / max(m["gt_auto_count"], 1)) * 100 if m["gt_auto_count"] else 0.0
    func = (m["func_correct"] / max(m["pred_auto_on_gt"], 1)) * 100 if m["pred_auto_on_gt"] else 0.0
    full = (m["fully_correct"] / total) * 100 if total else 0.0
    fail_pct = (unique_fails / total) * 100 if total else 0.0

    print("\n" + "=" * 80)
    print("FINAL AUDIT REPORT")
    print("-" * 80)
    print(f"1. Classification accuracy :  {m['correct_cls']} / {total}  ({cls_acc:.2f}%)")
    print(f"2. Automatable coverage    :  {m['pred_auto_on_gt']} / {m['gt_auto_count']}  ({cov:.2f}%)")
    print(f"3. Functional accuracy     :  {m['func_correct']} / {m['pred_auto_on_gt']}  ({func:.2f}%)")
    print(f"4. Fully correct           :  {m['fully_correct']} / {total}  ({full:.2f}%)")
    print(f"\n   TOTAL FAILURES          :  {unique_fails} / {total}  ({fail_pct:.2f}%)")
    print(f"   TRIVIAL N-GRAMS         :  {len(trivial_ngrams)}")
    print("=" * 80)

    final_payload = {
        "summary_metrics": {
            "total_items": total,
            "classification_accuracy": f"{m['correct_cls']} / {total} ({cls_acc:.2f}%)",
            "automatable_coverage": f"{m['pred_auto_on_gt']} / {m['gt_auto_count']} ({cov:.2f}%)",
            "functional_accuracy": f"{m['func_correct']} / {m['pred_auto_on_gt']} ({func:.2f}%)",
            "fully_correct": f"{m['fully_correct']} / {total} ({full:.2f}%)",
            "unique_failed_count": unique_fails,
            "trivial_ngrams_count": len(trivial_ngrams),
        },
        "scoring_configuration": {
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "string_overlap_threshold": STRING_OVERLAP_THRESHOLD,
            "structure_similarity_threshold": STRUCTURE_SIMILARITY_THRESHOLD,
            "crystal_similarity_threshold": CRYSTAL_SIMILARITY_THRESHOLD,
            "trivial_ngram_percentile": TRIVIAL_NGRAM_PERCENTILE,
            "max_trivial_ngrams": MAX_TRIVIAL_NGRAMS,
            "notes": {
                "ast_mode": "tree-sitter PowerShell AST (Airbus grammar) linearized to structural tokens",
                "crystal_mode": "CrystalBLEU-style trivial n-gram filtering before sequence similarity, not canonical CrystalBLEU",
            },
        },
        "failure_breakdown": {
            "classification_failures": m["fail_cls"],
            "coverage_failures": m["fail_coverage"],
            "logic_failures": m["fail_func"],
        },
        "detailed_failure_audit": f_logs,
    }

    with open(report_path, "w") as f:
        json.dump(final_payload, f, indent=2)

    print(f"\nFull report saved -> {report_path}", flush=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        required=True,
        help="Path to the LoRA adapter (final_model/)"
    )
    ap.add_argument(
        "--test-jsonl",
        required=True,
        help="Path to final_test_integrated.jsonl"
    )
    ap.add_argument(
        "--train-jsonl",
        default=None,
        help="Optional training JSONL used to derive trivial n-grams for CrystalBLEU-style filtering"
    )
    ap.add_argument(
        "--results-dir",
        default="final_eval_results",
        help="Output directory (default: final_eval_results)"
    )
    args = ap.parse_args()
    eval_model(args.model_dir, args.test_jsonl, args.results_dir, args.train_jsonl)        