#!/usr/bin/env python3
import os

# --- MANDATORY: SET BEFORE TORCH/TRANSFORMERS IMPORTS ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/mnt/lustre/koa/scratch/demersn/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/lustre/koa/scratch/demersn/hf_cache"

from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
BASE_OUT_DIR = "/mnt/lustre/koa/scratch/demersn/final_training_results"

# Updated to use STIG-level splits you generated
TRAIN_FILE = "/home/demersn/koa_scratch/final_train_integrated.jsonl"
EVAL_FILE = "/home/demersn/koa_scratch/final_val_integrated.jsonl"

MAX_SEQ_LEN = 4096

# Explicit HF token
HF_TOKEN = "THE_HUGGING_FACE_TOKEN_HERE"


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if is_distributed() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def barrier():
    if dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_string(value):
    if not is_distributed():
        return value
    obj_list = [value] if is_main_process() else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def get_precision_config():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"compute_dtype": torch.bfloat16, "bf16": True, "fp16": False}
    return {"compute_dtype": torch.float16, "bf16": False, "fp16": True}


def find_latest_resumable_run(base_dir: Path):
    latest_run = None
    latest_ckpt = None

    if not base_dir.exists():
        return None, None

    candidate_dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    candidate_dirs = sorted(candidate_dirs, key=lambda p: p.stat().st_mtime, reverse=True)

    for run_dir in candidate_dirs:
        ckpt = get_last_checkpoint(str(run_dir))
        if ckpt:
            latest_run = run_dir
            latest_ckpt = ckpt
            break

    return latest_run, latest_ckpt


def get_or_create_run_dir():
    base_dir = Path(BASE_OUT_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)

    selected_run_dir = None
    selected_ckpt = None

    if is_main_process():
        selected_run_dir, selected_ckpt = find_latest_resumable_run(base_dir)

        if selected_run_dir is not None:
            print(f"Found latest resumable run: {selected_run_dir}")
            print(f"Found latest checkpoint: {selected_ckpt}")
        else:
            run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_id = os.environ.get("SLURM_JOB_ID", "nojid")
            run_name = f"run_{job_id}_{run_stamp}"
            selected_run_dir = base_dir / run_name
            print(f"No resumable run found. Creating new run: {selected_run_dir}")

        selected_run_dir.mkdir(parents=True, exist_ok=True)
        (selected_run_dir / "final_model").mkdir(parents=True, exist_ok=True)
        (selected_run_dir / "logs").mkdir(parents=True, exist_ok=True)

        latest_link = base_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            try:
                latest_link.unlink()
            except Exception:
                pass
        try:
            latest_link.symlink_to(selected_run_dir, target_is_directory=True)
        except Exception:
            pass

    barrier()

    run_dir = Path(broadcast_string(str(selected_run_dir)))
    resume_ckpt = broadcast_string(selected_ckpt)

    return run_dir, resume_ckpt


def main():
    setup_distributed()

    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    prec = get_precision_config()

    run_dir, resume_checkpoint = get_or_create_run_dir()
    checkpoints_dir = run_dir
    final_model_dir = run_dir / "final_model"
    logs_dir = run_dir / "logs"

    if is_main_process():
        print(f"Precision config: {prec}")
        print(f"WORLD_SIZE={world_size}")
        print(f"RANK={rank}")
        print(f"LOCAL_RANK={local_rank}")
        print(f"Run directory: {run_dir}")
        print(f"Checkpoints directory: {checkpoints_dir}")
        print(f"Final model directory: {final_model_dir}")
        print(f"Resume checkpoint: {resume_checkpoint if resume_checkpoint else 'NONE'}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        outputs = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        outputs["labels"] = [
            token_id if token_id != tokenizer.pad_token_id else -100
            for token_id in outputs["input_ids"]
        ]
        return outputs

    raw_ds = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "eval": EVAL_FILE,
        },
    )

    tokenized_ds = raw_ds.map(
        tokenize_function,
        remove_columns=raw_ds["train"].column_names,
        desc="Pre-tokenizing with Chat Template",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=prec["compute_dtype"],
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=prec["compute_dtype"],
        token=HF_TOKEN,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        bf16=prec["bf16"],
        fp16=prec["fp16"],
        report_to=[],
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        logging_dir=str(logs_dir),
        ddp_find_unused_parameters=False if is_distributed() else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        peft_config=lora,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["eval"],
    )

    if is_main_process():
        print("Starting training...")

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    barrier()

    if is_main_process():
        print("Training complete. Saving final model...")
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))

        with open(run_dir / "run_info.txt", "w") as f:
            f.write(f"MODEL_NAME={MODEL_NAME}\n")
            f.write(f"WORLD_SIZE={world_size}\n")
            f.write(f"RUN_DIR={run_dir}\n")
            f.write(f"CHECKPOINTS_DIR={checkpoints_dir}\n")
            f.write(f"FINAL_MODEL_DIR={final_model_dir}\n")
            f.write("SAVE_STRATEGY=steps\n")
            f.write("SAVE_STEPS=10\n")
            f.write("LOGGING_STEPS=10\n")
            f.write(f"RESUMED_FROM={resume_checkpoint if resume_checkpoint else 'NONE'}\n")

        print("All done.")

    cleanup_distributed()


if __name__ == "__main__":
    main()