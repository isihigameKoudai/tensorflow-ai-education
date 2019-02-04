# tensorflow-ai-education

## 概要

TensorFlow.js を使った AI の入門キットです。Tensorflow だけでなく機械学習に関する基礎的な内容や、CNN や RNN などのアルゴリズムなど人工知能を取り扱っていくために体系的な知識群を掲載しています。

## AI （人工知能）概要

Artificial Intelligence の略。人工知能は**機械学習**と言う言葉に置き換えることが出来、言葉の通り機会が自律的に学習していくアルゴリズムやそれらのネットワークのことをさす。

### AI の種類

大きく分けると **Artificial General Intelligence(AGI)** と、 **Growing Artificial Intelligence(GAI)** に分けられる。

#### Artificial General Intelligence

**特化型人工知能（AGI）** とも呼ばれ、ある特定の分野のみに特化して能力を発揮する人工知能のこと。ある分野に特化することで人間を超えるほどの能力を発揮することがある。

#### Growing Artificial Intelligence

**汎用型人工知能（GAI）** は、異なる分野に置いて複雑な問題を解決する人工知能のこと。

#### 弱い人工知能(AI)・強い人工知能(AI)

**弱い人工知能** とは、ある枠の中だけで考える人工知能のこと。特定の範囲内の中だけで活動することで人間を超えるような結果も出すことがあり、そう言った意味で**特化型人工知能**に似ているが、これに関してはプログラムされたことしか出来ない。役割としては、人間の補佐。補助的なものが多い。

**強い人工知能** とは、**自己学習・自己フィードバック**はもちろんのこと自律的に思考し、ある特定の分野に縛られることなく活動する人口知能のこと。
こちらはみなさんが想像するような人間のように認識・思考・フィードバック、推論などしながら自己学習していくタイプ。先ほどの GAI の別の読み方と捉えることも出来る。

### AI の 4 つのレベル

| レベル   | 基準                                                                    |
| -------- | ----------------------------------------------------------------------- |
| レベル４ | 自分で判断基準を設計し、 対応パターンで使用する特徴量を自力で出していく |
| レベル３ | 対応パターンを自動的に学習し、基準を改善しながら判断していく            |
| レベル２ | 対応パターンがかなり多い、判断力が求められる                            |
| レベル１ | 単純な制御プログラム                                                    |

### 機械学習とその種類について
#### 機械学習（Machine Learning）
機械学習とは人間が自然と行なっているの脳の仕組みや機能と同等の動きをコンピュータで実現する技術と手法のこと

#### 機械学習の種類
- 教師あり学習
- 教師なし学習
- 転移学習
- 強化学習
- ニューラルネットワーク
- 深層学習（Deep Learning）
- CNN（Convolutional Neural Network）
- RNN（Recurrent Neural Network：再帰型ニューラルネットワーク）
- LSTM（Long Short-Term Memory）
- GAN (Generative Adversarial Network: 敵対的生成ネットワーク)

#### 教師あり学習（Supervised Learning）・・・回帰・分類
教師あり学習は人間が学習させた大量の「入力」と「正解データ」の特徴量を元に、未知のデータがなんであるかを判断していくような機械学習のこと。主に「回帰」と「分類」に用いられ、画像の判断や手書き文字の認識などによく使われている。

#### 教師なし学習（Unsupervised Learning）
教師なし学習は「正解データ」のない機械学習で、大量のデータをインプットさせるがそこに答えはなくどういった特徴量があるかを導き出す学習のこと。主に「クラスタリング」や「次元削減」に使われます。クラスタリングはバラバラに散らばったデータから関係性を学習しグルーピングします。次元削減は、多次元要素からなるデータの集まりがもつ特徴量を見つけ、本来あった次元を減らしていきます。

#### オートエンコーダー
オートエンコーダーとはNNの一種で、データから特徴量を抽出し情報量を少なくする「次元削減」をコアとしている。

#### 転移学習
　ある領域で学習させたモデルを他の領域に応用させる技術。また、通常学習させなければいけないデータよりもかなり少ないデータ量で高い精度を出すことができます。

#### 強化学習
強化学習は、望ましい答えを得られた場合に「報酬」を与えることにより、その報酬を得るような行動をしていくスタイルの機械学習です。囲碁などで使われることがあり、一手には判断基準がないが「勝つ」か「負ける」かという所に価値があり、そのために出来るだけ勝つことを達成するための行動をとるようになって行きます。一見、その一つの動作自体の影響力や価値は無いように見えますが、その最終地点の「勝つ・負ける」に置いて価値が置かれている場合、「勝つ」に寄せていくためにどんな一手を打つか、その基準が報酬になります。その報酬を得るためには、どう行動すれば良いか学習していき、最終的に勝つべくして行動します。

#### ニューラルネットワーク（パーセプトロン、CNN,RNN）
ニューロンと呼ばれる人間の脳内にある神経細胞によって構築されている神経回路網をモデル化・数式化したものをニューラルネットワーク（以下、NN）と呼ぶ。NNの内部構造は大きく分けると、入力層・隠れ層・出力層の３つのパーセプトロンから成っている。パーセプトロンとは複数の入力から一つの出力を行う関数で、出力値は０か１かの２値しか返さない関数となっている。それに対しNNはシグモイド関数と呼ばれる連続した関数軍によって変化する出力値をだす。

#### 深層学習（Deep Learning）
深層学習は前述のNNの隠れ層を何層にも重ねたネットワークのこと。

#### CNN（Convolutional Neural Network：畳み込みニューラルネットワーク）
CNNは深層学習の一種で「畳み込み層」と「プーリング層」の二つの幾重にも重なった層からなる隱れ層を持つのが特徴。畳み込み層ではフィルタの適用によって「特徴マップ」と呼ばれるデータの特徴的な部分を抽出します。プーリング層では畳み込み層で得られた特徴量を元に縮小したマップを生成します。つまりは、データをどんどん抽象化していくことがCNNの得意とすることで、主に画像認識や解析に使われることが多いです。

#### RNN（Recurrent Neural Network：再帰型ニューラルネットワーク）
CNNが取り扱っていた２次元的なものではなく、RNNでは「時系列」をコンテキストに含むNNで、一連のデータに流れを持たせていることが特徴である。隠れ層で扱ったデータを再び隠れ層にインプットすることで、前のデータの状態を踏まえた上でデータを処理します。主に音声や言語解釈で使われることが多く、文章や会話の理解・文脈の抽出などで使われたり、過去のデータを踏まえて未来のデータを予測する天気予報のようなところでも使われます。

#### LSTM（Long Short-Term Memory）
RNNの一種ですが、こちらは長期的な依存関係を学習するモデルです。隠れ層をLSTM blockと呼ばれるメモリと３つのゲートをもつユニットに置き換えるだけです。

#### GAN (Generative Adversarial Network: 敵対的生成ネットワーク)
生成ネットワーク（Generator）と識別ネットワーク（Discriminator）から成る教師なし学習のネットワークの一つで、Generaterは出来るだけ本物に近づけた偽物を生成し、それをDiscriminatorが強力に判断していく、というものを繰り返していきます。機械自身が仮定と否定を繰り返し限りなく答えに近くという、近年注目されている学習です。

## Tensorflow.js 入門

### 基本概念

#### Tensor

tensor とは多次元配列のことで、tensorflow の名前の通り多次元配列を中心に取り扱っていきます。tensor 自体は shape と呼ばれる、配列の形となるデータ構造を持ちます。基本的にtensorはイミュータブルなので、後からデータを書き換えることが出来ないですが、品質の担保に繋がります。

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

// 低次元配列を作るときは可読性を上げるため、tf.scalar, tf.tensor1d, tf.tensor2d, tf.tensor3d と tf.tensor4dを推奨します。
const c = tf.tensor2d([[1.0, 2,0],[3.0, 4.0]])
c.print()
// [[1 , 2],
//  [3, 4]]

// 全て0の配列を作ることも簡単にできます。
const zeros = tf.zeros([3, 4]);
zeros.print();
//   [[0, 0, 0, 0],
//    [0, 0, 0, 0],
//    [0, 0, 0, 0]]

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
Opsはデータを操作するためのもので、tesor自体はイミュータブルなので新しいtensorを返す形で配列操作を行います。
```
const d = tf.tensor2d([[5.0, 1.0], [2.0, 4.0]]);
const d_squared = d.square();
d_squared.print();
// Output: [[25, 1 ],
//          [4, 16]]
```

#### Model と Layer

#### メモリ管理

## 参考

https://qiita.com/tomomoto/items/b3fd1ec7f9b68ab6dfe2
http://gagbot.net/machine-learning/ml4
http://tensorflow.classcat.com/
https://qiita.com/tomomoto/items/b3fd1ec7f9b68ab6dfe2
http://gagbot.net/machine-learning/ml4
https://qiita.com/KojiOhki/items/89cd7b69a8a6239d67ca

### AI 全般  
https://www.sejuku.net/blog/7290
https://deepinsider.jp/
https://thinkit.co.jp/story/2015/09/09/6399
https://qiita.com/yoshizaki_kkgk/items/55b67daa25b058f39a5d

### おすすめ書籍  
https://qiita.com/tani_AI_Academy/items/4da02cb056646ba43b9d
https://www.data-artist.com/contents/ai-books.html

### LSTM
https://qiita.com/dojineko/items/ae7393dc83fb1f5fb0a4
https://qiita.com/t_Signull/items/21b82be280b46f467d1b
https://qiita.com/KojiOhki/items/89cd7b69a8a6239d67ca

### Tensorflow.js
http://tensorflow.classcat.com/2018/04/03/tensorflow-js-tutorials-core-concepts/
https://deepinsider.jp/tutor/introtensorflow
https://www.sejuku.net/blog/46586
https://qiita.com/yukagil/items/ca84c4bfcb47ac53af99
https://mizchi.hatenablog.com/entry/2018/10/28/211852
https://mizchi.hatenablog.com/entry/2018/10/01/135805


