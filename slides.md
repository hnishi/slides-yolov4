YOLOv4
===

YOLOv4: Optimal Speed and Accuracy of Object Detection

- Date: 2020-04-26
- Speaker: Hiroshi Nishigami
- Links:
  - pdf: https://arxiv.org/pdf/2004.10934.pdf
  - abs: https://arxiv.org/abs/2004.10934
  - github: https://github.com/AlexeyAB/darknet

***

## はじめに

- yolov3 の著者 pjreddie (Joseph Redmon) が出した論文ではない
- pjreddie は CV の研究から引退。
  - [軍事利用やプライバシーの問題を無視できなくなったからだとか。](https://twitter.com/pjreddie/status/1230524770350817280?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1253524202528731136&ref_url=https%3A%2F%2Fblog.seishin55.com%2Fentry%2F2020%2F05%2F16%2F183132)
- [この論文 yolov4 の first author は、 darknet を fork して開発を続けていた人。](https://github.com/AlexeyAB/darknet)
  - [本人のコメント](https://twitter.com/alexeyab84/status/1264188352271613952)

---

## おさらい

- [YOLO](https://pjreddie.com/darknet/yolo/) とは
  - You only look once (YOLO): [it looks for the image/frame only once and able to detect all the objects in the image/frame](https://medium.com/@rajansharma9467/yolo-you-only-look-once-2afdba1f6a32)
  - リアルタイム物体検出 (object detection) システム
  - 昨今の、object detection のスタンダードな手法

---

- [darknet](https://pjreddie.com/darknet/) とは
  - C で書かれたオープンソースのニューラルネットワークのフレームワーク
  - 構築済み (学習済み) の yolo series が利用できる

***
***

## 概要

- 最先端のテクニック・手法を (ある程度の仮説を立てながら) 総当たりで実験し、良いものを採用するための実験を行った (for single GPU)
- 性能が良かった組み合わせを採用して、YOLOv4 として提案
- 既存の高速(高FPS)のアルゴリズムの中で、最も精度が良い手法
- YOLOv3 よりも精度が高く、EfficientDet よりも速い

---

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-24-01-30-56.png" style="background:none; border:none; box-shadow:none;">

---

### AP: mean average precision (mAP)

- [averaged across all 10 IoU thresholds and all 80 categories](http://cocodataset.org/#detection-eval)

***
***

## 検証内容

1. Influence of different features on Classifier training
2. Influence of different features on Detector training
3. Influence of different backbones and pretrained weightings on Detector training
4. Influence of different mini-batch size on Detector training

---

## 手法探索とチューニング

- モデルアーキテクチャ
  - backbone: 画像の特徴抽出の役割
  - neck: backboneから受けた特徴マップをよしなに操作して、よりよい特徴量を生み出す
  - head: クラス分類やbbox(物体を囲む四角形)の位置を予測する
- Bag of freebies
  - 学習上の工夫
- Bag of specials
  - 少ないコスト(推論時間や計算リソース)で大きな精度向上ができるもの

---

### 物体検出器のアーキテクチャ

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-23-20-45-48.png" style="background:white; border:none; box-shadow:none;">

---

- backbone: 画像の特徴抽出の役割
- neck: backboneから受けた特徴マップをよしなに操作して、よりよい特徴量を生み出す
- head: クラス分類やbbox(物体を囲む四角形)の位置を予測する
  - 1-stage: 直接的に予測を行う
    - YOLO系列やSSD
    - 速度重視
  - 2-stage: 候補領域を出してから予測を行う
    - R-CNN系列
    - 精度重視

---

### ネットワークアーキテクチャ

Backbone: CSPDarknet53
Neck: SPP、PAN
Head: YOLOv3

---

#### Backbone: CSPDarknet53

- [CSPNet](https://arxiv.org/pdf/1911.11929.pdf)
  - 精度をあまり落とさずに、計算コストを省略するための手法
- CSPNet で提案される機構を Darknet53 (YOLOv3で使われているbackbone) に導入

---

#### Neck: SPP、PAN

  - [SPP (Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)](https://arxiv.org/abs/1406.4729)
    - 複数サイズの window でプーリングして特徴量を作り
    、受容野を広げることができる
  - [PAN (Path Aggregation Network for Instance Segmentation)](https://arxiv.org/pdf/1803.01534.pdf)

---

### 学習上の工夫 (Bag of freebies)と精度改善上の工夫 (Bag of specials)

- 活性化関数
  - Mish acrivation
  - bboxのregression loss
  - CIoU-loss
  - DIoU-NMS
- データオーグメンテーション
  - CutMix
  - Mosaic data augmentation
  - Self-Adversarial Training
- 正則化
  - DropBlock regularization
- 正規化
  - CmBN
- その他
  - Optimal hyper parameters
  - Cosine annealing scheduler
  - Class label smoothing

---

### Additional improvements

- Mosaic, a new data augmentation method
- Self-Adversarial Training (SAT)
- Genetic algorithmを使ったハイパーパラメータのチューニング
- modified SAM, modified PAN, and Cross mini-Batch Normalization (CmBN)

```note
Mosaic と SAT は著者らによって新しく提案された Data Augmentation の手法
```

---

#### Mosaic

4 つの画像を混ぜる

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-22-52-21.png" style="background:white; border:none; box-shadow:none;">

---

#### SAT: Self-Adversarial Training (SAT)

- 1度、network weights の代わりには元々の画像を更新 (self-adversarial attack)
- 2度目、この修正された画像に対して通常の学習を行う

---

#### Modified SAM

spatial-wise attention から pointwise attention へ修正

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-23-17-57.png" style="background:white; border:none; box-shadow:none;">

---

### Modified PAN

[PAN (Path Aggregation Network for Instance Segmentation)](https://arxiv.org/pdf/1803.01534.pdf)

shortcut connection を addition から concatenation に変更

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-23-24-12.png" style="background:white; border:none; box-shadow:none;">

---

#### Cross mini-Batch Normalization (CmBN)

[Cross-Iteration Batch Normalization (CBN) (2020/02/13 on arXiv)](https://arxiv.org/abs/2002.05712) をベースにして、改良を加えた

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-23-03-47.png" style="background:white; border:none; box-shadow:none;">

***
***

### 検証結果

---

#### Influence of different features on Classifier training

- data augmentation
  - bilateral blurring, MixUp, CutMix and Mosaic

---

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-23-47-04.png" style="background:white; border:none; box-shadow:none;">

---

- activations
  - Leaky-ReLU (as default), Swish, and Mish.

---

- [Mish (A Self Regularized Non-Monotonic Neural Activation Function)](https://arxiv.org/abs/1908.08681) は、昨年発表された連続な活性化関数 (2019年10月)

---

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-26-00-04-10.png" style="background:white; border:none; box-shadow:none;">

---

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-23-48-51.png" style="background:white; border:none; box-shadow:none;">

---

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-25-23-50-34.png" style="background:white; border:none; box-shadow:none;">

---

CutMix, Mosaic, Label Smoothing, Mish の効果が大きい

---

#### Influence of different features on Detector training

---

- Loss algorithms for bounded box regression
  - GIoU, CIoU, DIoU, MSE

- 単純な bbox の MSE 誤差よりも、IoU ベースの損失関数 ([CIoU-loss: Nov 2019](https://arxiv.org/abs/1911.08287)) の方が良かった

---

<img src="https://raw.githubusercontent.com/hnishi/slides-yolov4/master/attachments/2020-05-26-00-14-20.png" style="background:white; border:none; box-shadow:none;">

***
***

### 最終的に採用された手法

---

#### YOLOv4 アーキテクチャ

- Backbone: CSPDarknet53
- Neck: SPP , PAN
- Head: YOLOv3

---

#### Bag of Freebies (BoF) for backbone

(BoF: 学習時の手法)

- CutMix and Mosaic data augmentation
- DropBlock regularization
- Class label smoothing

---

#### Bag of Specials (BoS) for backbone

(BoS: 推論時のテクニック・追加モジュール)

- Mish activation
- Cross-stage partial connections (CSP)
- Multiinput weighted residual connections (MiWRC)

---

#### Bag of Freebies (BoF) for detector

- CIoU-loss
- CmBN
- DropBlock regularization
- Mosaic data augmentation
- Self-Adversarial Training
- Eliminate grid sensitivity
- Using multiple anchors for a single ground
truth
- Cosine annealing scheduler
- Optimal hyperparameters (genetic algorithm)
- Random training shapes

---

#### Bag of Specials (BoS) for detector

- Mish activation
- SPP-block
- SAM-block
- PAN path-aggregation block
- DIoU-NMS

***
***

## 議論、課題など

- detector の BoF を改善できる余地がある（future work）　
- FPSが70~130程度あるが、V100の強めのマシンであることには注意が必要（既存のモデルに対するパフォーマンスがよいことには変わりない）

---

エッジ領域である、Jetson AGX XavierはFPSが割と小さく、TensorRTに変換及び量子化等の対応が必要

Jetson AGX Xavier上で32FPS程度

Ref: https://github.com/AlexeyAB/darknet/issues/5386#issuecomment-621169646)

---

ちなみに

- NVIDIA Tesla V100
  - [販売価格(税別): ¥2,990,000](https://www.monotaro.com/p/3201/6094/?utm_medium=cpc&utm_source=Adwords&utm_campaign=246-833-4061_6466659573&utm_content=96539050923&utm_term=_419857551521__aud-368712506548:pla-879931900035&gclid=CjwKCAjwk6P2BRAIEiwAfVJ0rI_BDVoK7CUtr7mubZ5uS0cs-s8fLxzahnQYFKn_7w2sdZ3LkJb0fxoCd0AQAvD_BwE)

***
***

## 先行研究と比べて何がすごい？（新規性について）

- アーキテクチャ・手法が object detection 性能に与える影響を調査した
- 既存の手法よりも良い手法を提案した (YOLOv4)
  - YOLOv3 と同程度に速く、より高い精度
  - EfficientDet より速く、同程度の精度
- 速度重視で物体認識モデルを考えるのであれば、選択の筆頭候補ということになる

***
***

## 参考資料

- yolov4 の日本語解説資料
  - [[DL輪読会]YOLOv4: Optimal Speed and Accuracy of Object Detection](https://www.slideshare.net/DeepLearningJP2016/dlyolov4-optimal-speed-and-accuracy-of-object-detection-234027228)
  - [物体認識モデルYOLOv3を軽く凌駕するYOLOv4の紹介 - ほろ酔い開発日誌](https://blog.seishin55.com/entry/2020/05/16/183132)
- [英語解説記事](https://towardsdatascience.com/yolo-v4-optimal-speed-accuracy-for-object-detection-79896ed47b50)
- yolov3 architecture: [A Closer Look at YOLOv3](https://www.blog.dmprof.com/post/a-closer-look-at-yolov3)

***
