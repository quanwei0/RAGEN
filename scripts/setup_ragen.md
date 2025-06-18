# Manual Scripts to Setup Environment
```bash
conda create -n ragen python=3.9 -y
conda activate ragen


git clone git@github.com:ZihanWang314/ragen.git
cd ragen

pip install -e .
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "flash-attn<2.8.0" --no-build-isolation

git submodule init
git submodule update

pip install -r requirements.txt

cd verl
pip install -e .
cd ..

pip install "ray[default]" debugpy

```
