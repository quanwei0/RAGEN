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
- If you encounter issues with the httpx package, try `pip install httpx==0.23.3`

Install Conda (if needed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ragen
```


Install Java (if needed)
```bash
cd /opt
wget https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.1%2B12/OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz

sudo tar -xzf OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz
sudo mv jdk-21.0.1+12 jdk-21

export JAVA_HOME=/opt/jdk-21
export LD_LIBRARY_PATH=$JAVA_HOME/lib/server:$LD_LIBRARY_PATH
```

## Usage

Run the following scripts for PPO training:

> setup wandb configurations and CUDA_VISIBLE_DEVICES env variables at first 

```bash
cd RAGEN
conda activate ragen

bash train_ppo.sh
bash train_ppo_bilevel.sh
bash train_ppo_multi_turn.sh
```

Done:
- skip env tokens or not when assigning next_values and accumulating TD error in GAE calcualtion (see `/ragen/trainer/core_algos.py`)
- mask env tokens in critic update (see `/ragen/workers/critic/dp_critic.py`)
- compute return using adv + value or cumulative reward (see `/ragen/trainer/core_algos.py`)

To-Do:
- turn-level + token-level weighted advantage estimation