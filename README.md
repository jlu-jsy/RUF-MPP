# RUF-MPP

RUF-MPP is a graph neural network framework for multi-task molecular property prediction.

## 1. Clone

```bash
git clone https://github.com/jlu-jsy/RUF-MPP.git
cd RUF-MPP
```

## 2. Environment

```bash
conda env create -f environment.yml
conda activate ruf_mpp
```

## 3. Data

Put dataset files under `data/`:

```text
data/bbbp.csv
data/tox21.csv
data/sider.csv
data/clintox.csv
data/bace.csv
data/muv.csv
data/hiv.csv
data/toxcast.csv
```

Each CSV file should start with a `smiles` column:

```text
smiles,task1,task2,...
CCO,1,0,...
```

## 4. Train

Run without augmentation:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset bbbp \
  --device 0 \
  --aug none \
  --aug_ratio 0.0 \
  --epochs 500 \
  --batch_size 256 \
  --lr 0.0001
```

Run with rule-based augmentation:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset tox21 \
  --device 0 \
  --aug rule \
  --aug_ratio 1.0 \
  --epochs 500 \
  --batch_size 256 \
  --lr 0.0001
```

For ToxCast:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset toxcast \
  --device 0 \
  --aug rule \
  --aug_ratio 1.0 \
  --epochs 500 \
  --batch_size 128 \
  --lr 0.0001
```
