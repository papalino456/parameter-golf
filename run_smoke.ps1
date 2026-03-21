$ErrorActionPreference = 'Stop'

$env:RUN_ID = 'smoke'
$env:ITERATIONS = '1'
$env:WARMDOWN_ITERS = '0'
$env:WARMUP_STEPS = '0'
$env:VAL_LOSS_EVERY = '0'
$env:TRAIN_LOG_EVERY = '1'
$env:MAX_WALLCLOCK_SECONDS = '0'
$env:TORCHDYNAMO_DISABLE = '1'
$env:TRAIN_BATCH_TOKENS = '65536'
$env:VAL_BATCH_SIZE = '65536'
$env:TRAIN_SEQ_LEN = '512'
$env:TTT_BATCH_SIZE = '4'
$env:TTT_CHUNK_SIZE = '128'
$env:TTT_EVAL_SEQ_LEN = '512'
$env:SKIP_TTT_EVAL = '1'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$trainScript = Join-Path $root 'train_gpt.py'

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}

& $python -c "import os, runpy; [os.environ.pop(k, None) for k in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT')]; runpy.run_path(r'$trainScript', run_name='__main__')"
