# Multi-Turn RL for LLM Agents

## Installation

We build our codebase upon [RAGEN](https://github.com/RAGEN-AI/RAGEN).

To get started quickly, run the following scripts:
```bash
bash scripts/setup_ragen.sh
bash scripts/setup_webshop.sh
```

Note:
- For manual setup instructions, refer to `scripts/setup_ragen.md`.
- Ensure that the flash-attention package version is less than 2.8.0.
- A Java environment is required to run WebShop experiments.
- In `scripts/setup_webshop.sh`, dataset files need be downloaded from Google Drive. If gdown is not available, you can manually download the files using:

```bash
cd RAGEN
conda activate ragen

huggingface-cli download \
  --repo-type dataset \
  --local-dir ./external/webshop-minimal/webshop_minimal/data/full \
  quanwei0/webshop-minimal \
  items_ins_v2.json \
  items_shuffle.json
```


## Usage

Run the following scripts for PPO training:

> setup wandb configurations and CUDA_VISIBLE_DEVICES env variables at first 

```bash
cd RAGEN
conda activate ragen

bash train_ppo.sh
```