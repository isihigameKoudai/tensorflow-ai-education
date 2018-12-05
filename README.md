# tensorflow-ai-education

## 概要

TensorFlow.js を使った AI の入門キットです。

## AI （人工知能）概要

Artificial Intelligence の略。

### AI の種類

大きく分けると **Artificial General Intelligence(AGI)** と、 **Growing Artificial Intelligence(GAI)** に分けられる。

#### **Artificial General Intelligence**

**特化型人工知能（AGI）** とも呼ばれ、ある特定の分野のみに特化して能力を発揮する人工知能のこと。ある分野に特化することで人間を超えるほどの能力を発揮することがある。

#### **Growing Artificial Intelligence**

**汎用型人工知能（GAI）** は、異なる分野に置いて複雑な問題を解決する人工知能のこと。

**弱い人工知能(AI)・強い人工知能(AI)**  
**弱い人工知能** とは、ある枠の中だけで考える人工知能のこと。特定の範囲内の中だけで活動することで人間を超えるような結果も出すことがあり、そう言った意味で特化型人工知能に似ているが、これに関してはプログラムされてたことしか出来ない。役割としては、人間の補佐。補助的なものが多い。

**強い人工知能**  
**強い人工知能** とは、自己学習・自己フィードバックはもちろんのこと自律的に思考し、ある特定の分野に縛られることなく活動する人口知能のこと。
こちらはみなさんが想像するような人間のように認識・思考・フィードバック、推論などしながら自己学習していくタイプ。先ほどの GAI の別の読み方と捉えることも出来る。

### AI の 4 つのレベル

| レベル   | 基準                                                                    |
| -------- | ----------------------------------------------------------------------- |
| レベル４ | 自分で判断基準を設計し、 対応パターンで使用する特徴量を自力で出していく |
| レベル３ | 対応パターンを自動的に学習し、基準を改善しながら判断していく            |
| レベル２ | 対応パターンがかなり多い、判断力が求められる                            |
| レベル１ | 単純な制御プログラム                                                    |

## 用語集

- 機械学習（Machine Learning）
- 深層学習（Deep Learning）
- 学習モデル
- 転移学習
- ニューラルネットワーク

- 線形回帰と非線形回帰
- 学習済みモデル（ネットワークモデル）＝アルゴリズム＋トレーニングデータ
- 学習（learning）と推論（inference）
- 教師あり学習（Supervised Learning）・・・回帰・分類
- 教師なし学習（Unsupervised Learning）・・・クラスタリング・次元削減
- 強化学習
- オートエンコーダー
- CNN（Convolutional Neural Network：畳み込みニューラルネットワーク）・・・画像認識や音声認識
- RNN（Recurrent Neural Network：再帰型ニューラルネットワーク）・・・LSTM（Long Short-Term Memory）、自然言語や時系列データの識別・生成

## Tensorflow.js 入門

### 基本概念

#### Tensor

tensor とは多次元配列のことで、tensorflow の名前の通り多次元配列を中心に取り扱っていきます。tensor 自体は shape と呼ばれる、配列の形となるデータ構造を持ちます。

```js
// 2x3 Tensor
const shape = [2, 3] // 2 rows, 3 columns
const a = tf.tensor([4.0, 1.0, 2.0, 90.0, 80.0, 70.0], shape)
a.print()
// Output: [[4 , 1 , 2 ],
//          [90, 80, 70]]

const b = tf.tensor([[100, 200, 300], [10.0, 20.0, 30.0]])
b.print()
// [[100 , 200 , 300 ],
//  [10, 20, 30]]
```

#### Variables

Variables は Tensor の値によって初期化されますが、これははミュータブルです。assign メソッドにより値を書き換えることが出来ます。

```js
const initValues = tf.zeros([3])
const biases = tf.variable(initValues) // initialize biases
biases.print() // output: [0, 0, 0]

const updatedValues = tf.tensor1d([0, 1, 0])
biases.assign(updatedValues) // update values of biases
biases.print() // output: [0, 1, 0]
```

#### Ops

#### Model と Layer

#### メモリ管理

## 参考

http://tensorflow.classcat.com/

**AI 全般**  
https://www.sejuku.net/blog/7290
https://deepinsider.jp/
https://thinkit.co.jp/story/2015/09/09/6399
https://qiita.com/yoshizaki_kkgk/items/55b67daa25b058f39a5d

**おすすめ書籍**  
https://qiita.com/tani_AI_Academy/items/4da02cb056646ba43b9d
https://www.data-artist.com/contents/ai-books.html

**LSTM**  
https://qiita.com/t_Signull/items/21b82be280b46f467d1b

**Tensorflow.js**  
http://tensorflow.classcat.com/2018/04/03/tensorflow-js-tutorials-core-concepts/
https://deepinsider.jp/tutor/introtensorflow
https://www.sejuku.net/blog/46586
https://qiita.com/yukagil/items/ca84c4bfcb47ac53af99
https://mizchi.hatenablog.com/entry/2018/10/28/211852
https://mizchi.hatenablog.com/entry/2018/10/01/135805
