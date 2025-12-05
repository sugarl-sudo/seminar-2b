# 先進科学セミナー2B
実験コードリポジトリ
この環境はcaltというデータセット生成とモデル学習用のライブラリをベースにして作成しています
https://github.com/HiroshiKERA/calt-codebase


## 🚀 クイックスタート

CALT は `pip install calt-x` で導入できます。以下は全依存関係を含む最小セットアップ例です。

```bash
git clone https://github.com/sugarl-sudo/seminar-2b.git
cd seminar-2b
conda env create -f environment.yml
```

## 📦 データセット生成 (Chain-of-Thought 系タスク)
各スクリプトは `input : output` 形式で書き出し、`output_root/n=XX/...` という同じディレクトリ構造を維持します。

```bash
# ReLU 累積タスク
python scripts/dataset_generation/relu.py \
  --sequence-lengths 10 15 20 25 30 \
  --output-root data/relu/


# Square-mod タスク (逆順 split も同時生成)
python scripts/dataset_generation/square.py \
  --sequence-lengths 10 15 20 25 30 \
  --output-root data/square/


# 自己参照インデックスタスク (逆順 split も同時生成)
python scripts/dataset_generation/index.py \
  --sequence-lengths 13 31 \
  --m 2 \
  --output-root data/index/

```

サンプル数・シード・逆順/順列切り替え・入力値レンジなど主要フラグは、各 CLI の `--help` から調整可能です。

### 🔁 逆順データ生成の指定方法

- **ReLU / Square / Index 共通**: 何も指定しなければ正順 (`data.*`) と逆順 (`data-inv.*`) の両方が生成されます。逆順を省きたい場合は CLI に `--no-inverse` を付けてください。
- **ReLU + 順列指定**: `--permutation` を与えた場合のみ、逆順データは自動的にスキップされます（順列と逆順は同時指定不可のため）。

```bash
# 例: ReLU (正順+逆順を一括生成)
python scripts/dataset_generation/relu.py \
  --sequence-lengths 10 \
  --output-root data/relu/

# 例: Square と Index も同様に --no-inverse で逆順を省略可能
python scripts/dataset_generation/square.py \
  --sequence-lengths 10 \
  --output-root data/square/ \
  --no-inverse

python scripts/dataset_generation/index.py \
  --sequence-lengths 13 31 \
  --m 2 \
  --output-root data/index/ \
  --no-inverse
```

## 🧠 学習ジョブの実行

`scripts/train/train.py` は Hugging Face `Trainer` + CALT のデータローダを使った共通ランチャーです。訓練条件は `config/*.yaml` で管理しており、データパスを差し替えた逆順用設定 (`config/relu_inverse.yaml` など) も用意しています。

```bash
# 順方向 ReLU (n=10)
python scripts/train/train.py --config config/relu.yaml

# 逆順 ReLU (data-inv.* を使用)
python scripts/train/train.py --config config/relu_inverse.yaml

# Square / Index なども同様に
python scripts/train/train.py --config config/square.yaml
python scripts/train/train.py --config config/square_inverse.yaml
python scripts/train/train.py --config config/index.yaml
python scripts/train/train.py --config config/index_inverse.yaml
```

### オプション
- `--dryrun`: 1 エポック / 1000 サンプルに縮小してセットアップを素早く検証。
- `--no_wandb`: Weights & Biases へのログ送信を無効化。

その他タスク（Square, Index など）も同様に `config/` 以下の設定ファイルを複製し、`data.*` → `data-inv.*` に置き換えるだけで逆順データ学習の設定を追加できます。
