from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
WORKTREE_ROOT = ROOT.parent / "parameter-golf-worktrees" / "2026-03-27-round2"
LOG_DIR = ROOT / "logs" / "round2_seq_20260327"
RESULTS_TSV = ROOT / "results_round2.tsv"
BASELINE_LOG = ROOT / "logs" / "baseline_20260325.txt"
DATA_PATH = ROOT / "data" / "datasets" / "fineweb10B_sp1024"
TOKENIZER_PATH = ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"
RUN_TIMEOUT_SECONDS = 7_200


@dataclass(frozen=True)
class Experiment:
    slug: str
    branch: str
    description: str


EXPERIMENTS = [
    Experiment("exp13", "parameter-golf/2026-03-27-r2-exp13", "EMA controls with mixed int6 selection"),
    Experiment("exp12", "parameter-golf/2026-03-27-r2-exp12", "Bigram retune plus configurable EMA"),
    Experiment("exp15", "parameter-golf/2026-03-27-r2-exp15", "EMA schedule and sliding-eval controls"),
    Experiment("exp14", "parameter-golf/2026-03-27-r2-exp14", "Size rebalance for bigram, VE, and TTT"),
    Experiment("exp11", "parameter-golf/2026-03-27-r2-exp11", "Disable MTP heads by default"),
]


VAL_PATTERNS = [
    re.compile(r"legal_ttt_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    re.compile(r"final_int6_sliding_window_s64_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    re.compile(r"final_int6_sliding_window_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    re.compile(r"final_int6_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
    re.compile(r"final_int8_zlib_roundtrip_exact val_loss:[0-9.]+ val_bpb:([0-9.]+)"),
]
MEM_PATTERN = re.compile(r"peak memory allocated: (\d+) MiB")


def git(*args: str, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd or ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def parse_metrics(log_path: Path) -> tuple[float | None, float]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    val_bpb = None
    for pattern in VAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            val_bpb = float(matches[-1])
            break
    mem_match = MEM_PATTERN.findall(text)
    memory_gb = round((int(mem_match[-1]) / 1024.0) if mem_match else 0.0, 1)
    return val_bpb, memory_gb


def load_results() -> tuple[str, dict[str, list[str]]]:
    if not RESULTS_TSV.exists():
        return "commit\tval_bpb\tmemory_gb\tstatus\tdescription", {}
    lines = RESULTS_TSV.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "commit\tval_bpb\tmemory_gb\tstatus\tdescription", {}
    header = lines[0]
    rows: dict[str, list[str]] = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t", 4)
        if len(parts) == 5:
            rows[parts[0]] = parts
    return header, rows


def write_results(rows: dict[str, list[str]]) -> None:
    header = "commit\tval_bpb\tmemory_gb\tstatus\tdescription"
    ordered = sorted(rows.values(), key=lambda row: row[0])
    RESULTS_TSV.write_text(
        "\n".join([header, *("\t".join(row) for row in ordered)]) + "\n",
        encoding="utf-8",
    )


def upsert_result(commit: str, val_bpb: float, memory_gb: float, status: str, description: str) -> None:
    _, rows = load_results()
    rows[commit] = [
        commit,
        f"{val_bpb:.8f}",
        f"{memory_gb:.1f}",
        status,
        description,
    ]
    write_results(rows)


def ensure_baseline() -> tuple[str, float]:
    baseline_commit = git("rev-parse", "--short", "parameter-golf/2026-03-25-baseline-30m")
    baseline_bpb, baseline_memory = parse_metrics(BASELINE_LOG)
    if baseline_bpb is None:
        raise RuntimeError(f"Could not parse baseline val_bpb from {BASELINE_LOG}")
    upsert_result(
        baseline_commit,
        baseline_bpb,
        baseline_memory,
        "keep",
        "baseline 30m branch parameter-golf/2026-03-25-baseline-30m",
    )
    return baseline_commit, baseline_bpb


def run_experiment(exp: Experiment, baseline_bpb: float) -> tuple[str, float | None, float, str]:
    worktree = WORKTREE_ROOT / exp.slug
    commit = git("rev-parse", "--short", "HEAD", cwd=worktree)
    log_path = LOG_DIR / f"{exp.slug}.txt"
    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": f"round2_{exp.slug}",
            "DATA_PATH": str(DATA_PATH),
            "TOKENIZER_PATH": str(TOKENIZER_PATH),
            "TORCHDYNAMO_DISABLE": "1",
            "TRAIN_BATCH_TOKENS": "131072",
            "VAL_BATCH_SIZE": "65536",
            "VAL_LOSS_EVERY": "0",
            "MAX_WALLCLOCK_SECONDS": "1800",
            "EVAL_STRIDE": "0",
        }
    )
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting {exp.slug} {commit} :: {exp.description}", flush=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"# branch={exp.branch}\n# commit={commit}\n# description={exp.description}\n"
            "# env: TORCHDYNAMO_DISABLE=1 TRAIN_BATCH_TOKENS=131072 VAL_BATCH_SIZE=65536 "
            "VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=1800 EVAL_STRIDE=0\n\n"
        )
        handle.flush()
        try:
            proc = subprocess.run(
                [str(VENV_PYTHON), str(worktree / "train_gpt.py")],
                cwd=str(worktree),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT_SECONDS,
                check=False,
            )
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {RUN_TIMEOUT_SECONDS} seconds\n")
            return_code = -999
    val_bpb, memory_gb = parse_metrics(log_path)
    elapsed_min = (time.time() - start) / 60.0
    if return_code != 0 or val_bpb is None:
        status = "crash"
        score_for_tsv = 0.0
        upsert_result(commit, score_for_tsv, memory_gb, status, exp.description)
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] finished {exp.slug} status=crash "
            f"returncode={return_code} elapsed_min={elapsed_min:.1f}",
            flush=True,
        )
        return commit, None, memory_gb, status
    status = "keep" if val_bpb < baseline_bpb else "discard"
    upsert_result(commit, val_bpb, memory_gb, status, exp.description)
    delta = val_bpb - baseline_bpb
    print(
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] finished {exp.slug} status={status} "
        f"val_bpb={val_bpb:.8f} delta_vs_baseline={delta:+.8f} elapsed_min={elapsed_min:.1f}",
        flush=True,
    )
    return commit, val_bpb, memory_gb, status


def write_summary(
    baseline_commit: str,
    baseline_bpb: float,
    completed: list[tuple[Experiment, str, float | None, float, str]],
) -> None:
    summary_path = LOG_DIR / "summary.txt"
    lines = [
        f"baseline_commit={baseline_commit}",
        f"baseline_val_bpb={baseline_bpb:.8f}",
        "",
        "slug\tcommit\tval_bpb\tdelta_vs_baseline\tmemory_gb\tstatus\tdescription",
    ]
    sortable: list[tuple[float, str]] = []
    for exp, commit, val_bpb, memory_gb, status in completed:
        sort_key = val_bpb if val_bpb is not None else 999.0
        delta = (val_bpb - baseline_bpb) if val_bpb is not None else None
        delta_text = f"{delta:+.8f}" if delta is not None else "NA"
        val_text = f"{val_bpb:.8f}" if val_bpb is not None else "CRASH"
        row = "\t".join(
            [
                exp.slug,
                commit,
                val_text,
                delta_text,
                f"{memory_gb:.1f}",
                status,
                exp.description,
            ]
        )
        sortable.append((sort_key, row))
    lines.extend(row for _, row in sorted(sortable, key=lambda item: item[0]))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not VENV_PYTHON.exists():
        raise FileNotFoundError(f"Missing venv python: {VENV_PYTHON}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    baseline_commit, baseline_bpb = ensure_baseline()
    completed: list[tuple[Experiment, str, float | None, float, str]] = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp, baseline_bpb)
        completed.append((exp, *result))
        write_summary(baseline_commit, baseline_bpb, completed)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] all experiments complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
