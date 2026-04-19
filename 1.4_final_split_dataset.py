#!/usr/bin/env python3
import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def infer_stig_shortname_from_user(msg_content: str) -> str:
    """Extract STIG shortname from build_user() formatted content."""
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

    # Fallback: single token on first line
    first_line = msg_content.splitlines()[0].strip() if msg_content.splitlines() else ""
    if first_line and " " not in first_line and len(first_line) <= 20:
        return first_line

    return ""


def get_stig_from_record(o: dict) -> str:
    """Pull STIG shortname from SFT messages format."""
    for m in (o.get("messages") or []):
        if m.get("role") == "user":
            stig = infer_stig_shortname_from_user(m.get("content", ""))
            if stig:
                return stig
    return ""


def main():
    ap = argparse.ArgumentParser(
        description="STIG-level splitter for SFT JSONL — prevents leakage across splits."
    )
    ap.add_argument("--in_jsonl",  required=True, help="Input JSONL (sft_training.jsonl)")
    ap.add_argument("--out_dir",   required=True, help="Output directory")
    ap.add_argument("--seed",  type=int,   default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val",   type=float, default=0.15)
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_dir = Path(args.out_dir)

    print("Input: ", in_path.resolve())
    print("Output:", out_dir.resolve())

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path.resolve()}")

    if not (0 < args.train < 1) or not (0 < args.val < 1) or (args.train + args.val) >= 1:
        raise ValueError("Require: 0<train<1, 0<val<1, train+val < 1.")

    out_dir.mkdir(parents=True, exist_ok=True)

    by_stig = defaultdict(list)
    n_records = 0
    n_missing_stig = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            stig = get_stig_from_record(o)
            if not stig:
                n_missing_stig += 1
                stig = "__unknown__"
            by_stig[stig].append(o)
            n_records += 1

    if n_missing_stig:
        print(f"WARNING: {n_missing_stig} records had no detectable STIG shortname -> bucketed as __unknown__")

    stigs = list(by_stig.keys())
    random.seed(args.seed)
    random.shuffle(stigs)

    n_stigs  = len(stigs)
    n_train  = int(n_stigs * args.train)
    n_val    = int(n_stigs * args.val)

    train_stigs = set(stigs[:n_train])
    val_stigs   = set(stigs[n_train:n_train + n_val])
    test_stigs  = set(stigs[n_train + n_val:])

    def write(filename, stigset):
        path = out_dir / filename
        count = 0
        with path.open("w", encoding="utf-8") as out:
            for s in stigset:
                for o in by_stig[s]:
                    out.write(json.dumps(o, ensure_ascii=False) + "\n")
                    count += 1
        return path, count

    p_train, c_train = write("final_train_integrated.jsonl", train_stigs)
    p_val,   c_val   = write("final_val_integrated.jsonl",   val_stigs)
    p_test,  c_test  = write("final_test_integrated.jsonl",  test_stigs)

    (out_dir / "split_stigs.json").write_text(json.dumps({
        "seed":   args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": round(1.0 - args.train - args.val, 4)},
        "counts": {
            "records": n_records,
            "stigs":   n_stigs,
            "train_records": c_train,
            "val_records":   c_val,
            "test_records":  c_test,
        },
        "train": sorted(train_stigs),
        "val":   sorted(val_stigs),
        "test":  sorted(test_stigs),
    }, indent=2), encoding="utf-8")

    print(f"\nRecords read:   {n_records}")
    print(f"Unique STIGs:   {n_stigs}")
    print(f"STIG split:     train={len(train_stigs)}  val={len(val_stigs)}  test={len(test_stigs)}")
    print(f"\nWrote {c_train:>5} records -> {p_train}")
    print(f"Wrote {c_val:>5} records -> {p_val}")
    print(f"Wrote {c_test:>5} records -> {p_test}")


if __name__ == "__main__":
    main()