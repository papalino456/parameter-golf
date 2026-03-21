# autoresearch

This is an experiment to have the LLM do its own research, while staying aligned with the OpenAI Parameter Golf challenge constraints.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date. If you use a branch, prefer something like `parameter-golf/<tag>` and make sure it does not already exist.
2. **Read the in-scope files**: always read these files for current context before changing anything:
   - `AGENTS.md` — repo-specific operating contract.
   - `README.md` — current challenge rules, leaderboard, and latest SOTA.
   - `data/README.md` — dataset and tokenizer workflow.
   - `train_gpt.py` — root CUDA sandbox for quick prototyping.
3. **Respect the repo workflow**: use the root `train_gpt.py` only as a sandbox. Do not treat root-script edits as submission artifacts.
4. **Verify challenge data exists**: make sure the published FineWeb export and tokenizer are present under the canonical layout:
   - `data/datasets/<dataset_name>/`
   - `data/tokenizers/`
   - `data/manifest.json`
   If not, use the published downloader, typically `python data\cached_challenge_fineweb.py --variant sp1024` or for smoke-only setup `python data\cached_challenge_fineweb.py --variant sp1024 --train-shards 1`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once setup looks good, begin experimentation.

## Experimentation

The primary objective is to minimize `val_bpb` on the fixed `fineweb_val_*` split while staying challenge-legal.

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train_gpt.py`.

**What you CAN do:**
- Modify the root `train_gpt.py` — this is the only file you edit.
- Change model architecture, optimizer, hyperparameters, training loop, learning rate schedule, context length, compression strategy, etc. as long as the metric remains correct and the artifact remains reproducible.
- Explore non-record ideas too, provided they are clearly labeled and still respect artifact accounting when relevant.

**What you CANNOT do for challenge-legal artifacts:**
- Depend on external downloads, network calls, or training-data access during evaluation.
- Quietly change tokenizer or dataset assumptions without proving `val_bpb` remains correct.
- Treat the root script as a final submission artifact.
- Assume local RTX results are equivalent to final 8xH100 leaderboard behavior.


**Hard constraints to remember:**
- Primary metric: `val_bpb` on the fixed validation split.
- Counted code bytes + compressed model bytes must be `<= 16,000,000`.
- Leaderboard submissions must train in `<= 10` minutes on `8xH100` and evaluate in `<= 10` minutes on `8xH100`.
- For record submissions, beat the current SOTA by at least `0.005` nats with `p < 0.01` significance unless the change is purely a systems speedup.

**Data and tokenizer invariants:**
- Default apples-to-apples local comparisons should use the published `sp1024` setup with `VOCAB_SIZE=1024`.
- Validation always uses the full fixed `fineweb_val_*` split, even if training uses fewer shards.
- `--train-shards 1` is for smoke tests only; `python data\cached_challenge_fineweb.py --variant sp1024` is the normal local comparison setup.
- If you rebuild tokenizers or shards from published docs, preserve the published docs-cache lineage and keep proof such as `docs_selected.source_manifest.json` and `docs_sha256`.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run:** establish a baseline before trying new ideas.

## Recommended local workflow on this machine

This machine is Windows with CUDA / PyTorch, so prefer the CUDA path rather than `train_gpt_mlx.py`.

Useful commands:

```powershell
.\run_smoke.ps1
```

for a reduced-memory smoke test of the root script, or:

```powershell
$env:RUN_ID="baseline_rtx"
$env:VAL_LOSS_EVERY="200"
python -m torch.distributed.run --standalone --nproc_per_node=1 train_gpt.py
```

For controlled comparisons, keep overrides matched across runs where possible:
- `DATA_PATH`, `TOKENIZER_PATH`, `VOCAB_SIZE`
- `SEED`
- `ITERATIONS` or `MAX_WALLCLOCK_SECONDS`
- `VAL_LOSS_EVERY`
- ideally also `TRAIN_BATCH_TOKENS` and `TRAIN_SEQ_LEN` if memory allows

## Output and evidence

The exact log format can vary by artifact, but the important final evidence is the validation output, especially the `final_int8_zlib_roundtrip` lines because they reflect the compressed artifact actually being validated.

Track at least:
- final `val_bpb`
- final `val_loss`
- artifact bytes
- steps reached
- `ms/step` or effective throughput
- whether the script ran cleanly from the working folder
- whether the final roundtrip validation lines were emitted

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

Where:
1. `commit` is the short git hash for the experiment state
2. `val_bpb` is the achieved validation score, or `0.000000` for crashes
3. `memory_gb` is peak memory rounded to one decimal place, or `0.0` for crashes
4. `status` is `keep`, `discard`, or `crash`
5. `description` is a short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	12.0	keep	baseline
b2c3d4e	0.993200	10.2	keep	increase LR to 0.04
c3d4e5f	1.005000	8.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `parameter-golf/mar5` or `parameter-golf/mar5-gpu0`).

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train_gpt.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results from the log, including final `val_bpb` and any final roundtrip metrics.
6. If the log is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

## Promotion rules

if the results are promising, copy the counted implementation into a new dated folder under the appropriate `records/` track and test it from inside that folder.

Treat an artifact as locally promising only if it:
- passes a smoke test
- preserves correct validation behavior
- is competitive on matched local comparisons
- stays within the `16,000,000` byte limit

Before calling something submission-ready, confirm all of the following:
- it runs from inside its own `records/...` folder
- the final log includes the roundtrip validation lines
- the counted artifact fits within the size cap
- the folder includes `README.md`, `submission.json`, logs, and `train_gpt.py`
- the result is compared against the latest SOTA from root `README.md`
- record claims have enough seed evidence for significance

## Important limitations

- Local RTX comparisons are only for screening; they do not replace final 8xH100 validation.
- Do not compare artifacts directly if they use different tokenizer or data assumptions unless that difference is the point of the experiment.
- The root scripts are starting points, not the final location for the best submission artifacts.
- Do not overwrite an existing record folder; always create a new dated one.


The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!