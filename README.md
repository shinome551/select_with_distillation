# select_with_distillation

## 目的
全学習データを用いて学習したモデル（教師モデル）の出力をもとに、データの選択を行います.
データの選択は、そのサブセットを学習するモデル（生徒モデル）の精度が出来るだけ教師モデルに近づくように行います.
精度を維持しながらどれだけデータセットを削減できるか、というのが目標です.

## 手順
基本的には、教師モデルが出力する各データの交差エントロピー損失を指標とし、その降順に任意の数のデータを取り出すことで学習データのサブセットを構築します（select.py）.
しかし、そのままではサブセットのサイズが小さい（10000）場合に精度がガクッと落ちます。いわゆる過学習と呼ばれるものです.
そのため、知識の蒸留（knowledge distillation）[1]を用いて正則化を行います。教師モデルは既に存在しますから、使えるものは使いましょう(select_with_distillation.py).
蒸留を合わせて使うことで、サブセットのサイズが小さい場合でも精度の劣化は起こりません。ついでに教師モデルの精度を超える場合があることもわかります。この現象は蒸留ではよく見られるものですが、不思議ですね.
気が向いたら以下に実行結果を一覧するイカした表でも貼るつもりです.

## モデル
教師モデルにはGitHub上で公開されている学習済みのResnet18を使います([Link](https://github.com/chenyaofo/CIFAR-pretrained-models)).
後日自作のものに置き換えるかもしれません.
```
git clone https://github.com/chenyaofo/CIFAR-pretrained-models
```

## Reference
1. Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

[1]: https://arxiv.org/abs/1503.02531
