# Titans: テスト時に記憶を学習する

[![arXiv](https://img.shields.io/badge/arXiv-2501.00663-b31b1b.svg)](https://arxiv.org/abs/2501.00663)

*[English version here](README.md)*

このリポジトリは、Ali Behrouz、Peilin Zhong、Vahab Mirrokniによる論文[Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)の非公式PyTorch実装です。

**免責事項**: これは独立した再現実装であり、原著者との提携や承認を受けていません。この実装は論文に記載されている説明のみに基づいており、原著作の全ての側面を完全に再現できていない可能性があります。

## 概要

以下の概要は原論文の記述に基づいています。この実装は完全にテストされていないため、実際の動作は以下の説明と異なる場合があります。

Titansは、アテンション機構と再帰モデルの強みを組み合わせた新しいニューラルアーキテクチャです。歴史的コンテキストを記憶するニューラル長期記憶モジュールを導入し、モデルに以下の能力を与えます：

- 現在のコンテキストに注意を払いながら、長い過去の情報を活用する
- 2M以上のトークンのコンテキストウィンドウにスケールする
- 高速な並列化可能なトレーニングと効率的な推論を維持する

このアーキテクチャは従来のアプローチの限界に対処します：
- **再帰モデル**: データを固定サイズのメモリ（隠れ状態）に圧縮
- **アテンション機構**: コンテキストウィンドウ全体への注意を可能にするが、二次的なコストがかかる

Titansは二重記憶システムを導入します：
- **短期記憶**: 限られたコンテキストだが正確な依存関係モデリングを持つアテンション
- **長期記憶**: データを永続的に保存するための記憶能力を持つニューラルメモリ

## 実装状況

このリポジトリは、論文で説明されているMAC（Memory-Augmented Context）バリアントに焦点を当てたTitansアーキテクチャのPyTorch実装を提供しています。この実装は進行中であり、原著作の全ての詳細や性能特性を完全に再現していない可能性があります。

リポジトリには以下が含まれています：

- `modeling_titans_mac.py`: 論文の説明に基づいたTitans MACモデルの実装
- `configuration_titans_mac.py`: モデルの設定クラス
- `example.py`: モデル実装をテストするための基本的なサンプルスクリプト

**注意**: この実装は実験的なものであり、原著者の実装や結果に対して検証されていません。ユーザーは特定のユースケースに対するこのコードの動作と性能を検証する必要があります。

## 必要条件

```
torch==2.6.0+cu124
transformers==4.49.0
```

## 使用方法

以下は現在の実装の使用例です。これは基本的な例であり、モデルの動作が論文の説明と完全に一致しない可能性があることに注意してください：

```python
import torch
from modeling_titans_mac import TitansMACConfig, TitansMACForCausalLM

# 設定を作成
config = TitansMACConfig(
    num_tokens=10000,
    d_model=256,
    depth=2,
    num_heads=4,
    dim_head=64,
    segment_len=32,
    num_longterm_mem_tokens=4,
    persistent_size=8,
    neural_memory_segment_len=36,
    context_window=256,
)

# モデルを初期化
model = TitansMACForCausalLM(config)

# 入力を作成
input_ids = torch.randint(0, config.num_tokens, (1, 16))

# 順伝播
outputs = model(input_ids=input_ids)

# テキスト生成
# カスタム生成関数については example.py を参照
```

実装と使用方法の詳細については、`example.py`を参照してください。これは実験的な実装であるため、特定のニーズに合わせてパラメータを調整したりコードを修正したりする必要があるかもしれません。

## 引用

この実装が役立つと感じた場合は、原論文を引用してください：

```bibtex
@misc{behrouz2024titans,
      title={Titans: Learning to Memorize at Test Time},
      author={Ali Behrouz and Peilin Zhong and Vahab Mirrokni},
      year={2024},
      eprint={2501.00663},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## 免責事項

これはTitansアーキテクチャに対する学術的関心から作成された独立したプロジェクトです。この実装は論文に対する私の理解に基づいており、著者のオリジナル実装とは異なる可能性があります。私は原著者と提携しておらず、この作業は彼らによってレビューされていません。

問題を発見したり、改善のための提案がある場合は、issueを開くかプルリクエストを提出してください。
