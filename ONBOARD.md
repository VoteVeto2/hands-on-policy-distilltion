# Onboarding

## Set up the environment

Run these from a fresh parent directory (skip `mkdir`/`cd` if you're already in the project root):

```bash
mkdir on-policy-distillation
cd on-policy-distillation

git init
git clone https://github.com/huggingface/trl.git third_party/trl

uv venv .venv                  # or just `uv venv`, which defaults to .venv
source .venv/bin/activate

uv pip install -e third_party/trl
uv pip install -e "third_party/trl[vllm]"
uv pip install peft datasets accelerate
```

Notes:

- `-e` installs TRL in editable mode, so local edits in `third_party/trl` take effect without reinstalling.
- The `[vllm]` extra already includes the base install. vLLM needs Linux with a compatible GPU and CUDA.
- `uv pip` finds `.venv` automatically, so `source .venv/bin/activate` is optional. You can also prefix commands with `uv run`.
