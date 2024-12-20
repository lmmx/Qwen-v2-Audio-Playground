Qwen Audio

```
conda create -n qwa python=3.11 -y
conda activate qwa
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
uv pip install -e .
```
