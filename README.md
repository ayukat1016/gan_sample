# GANディープラーニング実装ハンドブック

![<img width="186" alt="GAN" src="https://user-images.githubusercontent.com/40778791/107626591-6120b500-6ca1-11eb-8598-9b7dad32a119.PNG"> (https://www.amazon.co.jp/gp/product/4798062294?pf_rd_r=ZTMA91YPE324B3QHMFNB&pf_rd_p=3d55ec74-6376-483a-a5a7-4e247166f80b&pd_rd_r=225c6ecb-5120-405a-b170-c214443ff320&pd_rd_w=kM9v5&pd_rd_wg=PJNj3&ref_=pd_gw_unk)

[GANディープラーニング実装ハンドブック](https://www.amazon.co.jp/gp/product/4798062294?pf_rd_r=ZTMA91YPE324B3QHMFNB&pf_rd_p=3d55ec74-6376-483a-a5a7-4e247166f80b&pd_rd_r=225c6ecb-5120-405a-b170-c214443ff320&pd_rd_w=kM9v5&pd_rd_wg=PJNj3&ref_=pd_gw_unk)

## 章の構成
- 第1章：生成モデル    
- 第2章：変分オートエンコーダ(VAE)
- 第3章：GANの基本モデル(DCGAN、CGAN、LSGAN)    
- 第4章：超解像(ESRGAN)    
- 第5章：ドメイン変換(pix2pix、CycleGAN)    
- 第6章：動画変換(Recycle-GAN)
- 第7章：StyleGAN、StyleGAN2
- 第8章：異常検知(AnoGAN、EfficientGAN)
- 第9章：3Dデータの生成(3D-α-WGAN-GP)
- Appendix：理論の補足

## ライブラリのバージョン
ライブラリは執筆時点のColabの最新バージョンになります。Colabのライブラリは定期的に更新するので、プログラム実行時にエラーが発生する場合はバージョンを戻して実行してください。
- torch:1.7.0
- torchvision:0.8.1
- pandas:1.1.5
- numpy:1.19.5
- matplotlib:3.2.2

## データセット
| 章 | モデル | データセット | ライセンス |取得元リンク|
|:-----------|:----------|:------------------------|:------------|:------------|
| 2 | AE、VAE | MNIST       | Creative Commons  |https://pytorch.org/vision/0.8/datasets.html |
| 3 | DCGAN、CGAN | MNIST       | Creative Commons  |https://pytorch.org/vision/0.8/datasets.html |
| 3 | LSGAN、DCGAN | Pet Dataset| Creative Commons  |https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| 4 | ESRGAN | Pet Dataset| Creative Commons  |https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| 5 | pix2pix、CycleGAN | photo2portrait | データセットの画像をインターネット等で公開したり、販売するのは禁止です。  |https://drive.google.com/file/d/1arF3guFms5tLiaIs8GtcV2dW0WAvrvLM/view?usp=sharing |
| 6 | Cycle GAN、Recycle-GAN | VidTIMIT Audio-Video Dataset| リンク先のLICENSEに利用時の注事事項の記載あり|https://conradsanderson.id.au/vidtimit/ |
| 7 | StyleGAN、StyleGAN2 | Endless Summer Dataset| データセットの画像をインターネット等で公開したり、販売するのは禁止です。 |https://drive.google.com/file/d/1LM4FtUltzS45PuFyfuSp3I8QdTD8Cu0F/view?usp=sharing |
| 8 | AnoGAN、EfficientGAN | Fruits 360 Dataset | Creative Commons  |https://data.mendeley.com/datasets/rp73yg93n8/1 |
| 9 | 3D-α-WGAN-GP | IXI Dataset| Creative Commons |http://brain-development.org/ixi-dataset/ |


## 学習時の注意点
| 章 | モデル | 注意点 |学習の目安時間 |
|:-----------|:------------|:------------|:------------|
| 3 | LSGAN、DCGAN | LSGANよりDCGANの方が猫っぽい画像を生成します。| 5時間程度|
| 4 | ESRGAN | デフォルト設定だとファイルはColabに保存されるので、Google Driveに保存したい場合は出力ファイルのパスの定義(output_dir)をGoogle Driveに変更してください。| 5～6時間程|
| 5 | pix2pix, CycleGAN | 特になし| 5〜8時間程度 |
| 6 | CycleGAN 、Recycle-GAN| 特になし| 半日から数日 |
| 7 | StyleGAN、StyleGAN2 | GPUは執筆時点で最速のP100を推奨（理想はV100）。| P100で2週間程度|
| 8 | AnoGAN、EfficientGAN、<br />EfficientGAN_L1 | AnoGAN, EfficientGANで生成精度が悪い場合、EfficientGAN_L1を使用してください。 | 2 ~ 3時間程度 |
| 9 | 3D-α-WGAN-GP | 特になし | 8時間程度|


## 生成画像の例

| 章 | モデル |説明| 生成画像 |
|:--|:---------|:------------|:------------------------------------------|
| 3 | DCGAN |サイズ128×128の猫画像を生成 | <img width="256" alt="fake_cat" src="https://user-images.githubusercontent.com/40778791/107382141-e8034f80-6b32-11eb-8029-563864814cfd.png"/> |
| 4 | ESRGAN | 低解像画像を入力し、超解像画像を生成 <br /><br />左側: 低解像画像 <br>中央: 本物画像 <br>右側: 生成画像（超解像画像）| ![image](https://user-images.githubusercontent.com/34574033/107609881-83a3d580-6c83-11eb-8cfb-acfd78c83b05.png) |
| 5 | CycleGAN | 肖像画を入力し、写真画像を生成 | ![0089](https://user-images.githubusercontent.com/20309500/107369273-eb8fda00-6b24-11eb-9e5b-623f1666403f.png) |
| 6 | Recycle-GAN |xx | ![A2B](https://user-images.githubusercontent.com/15444879/107294483-d1b4af80-6ab0-11eb-82b7-a44c322eb403.png) |
| 7 | StyleGAN2 |512×512の画像を生成|![2021 02 08_01](https://user-images.githubusercontent.com/21982866/107151130-52d25080-69a4-11eb-83ff-1d6d24c642cc.png)![2021 02 08_05](https://user-images.githubusercontent.com/21982866/107151201-a80e6200-69a4-11eb-9eaa-d2c6d485787d.png)![2021 02 08_09](https://user-images.githubusercontent.com/21982866/107151263-f7549280-69a4-11eb-8d2f-7ccdefaab869.png)|
| 8 | AnoGAN、EfficientGAN |サイズ96×96のほおずきの画像を生成<br /><br />左側: 入力画像 <br>中央: 生成画像 <br>右側: 差分画像 <br><br> 上段は本物画像を入力したため、本物と生成の差が小さく異常スコアが低い <br><br> 下段は異常画像を入力したため、生成画像は異常箇所を再現できず、異常スコアが高い | ![Screenshot from 2021-02-11 15-54-29](https://user-images.githubusercontent.com/42464037/107609609-cfa24a80-6c82-11eb-877d-988a000a593a.png)<br /><br />![Screenshot from 2021-02-11 15-34-19](https://user-images.githubusercontent.com/42464037/107609619-d5982b80-6c82-11eb-97e4-ca48f6290e9e.png) |
| 9 | 3D-α-WGAN-GP | サイズ64×64×64の3次元頭部MRIデータを生成 <br>3つの画像は1つの3次元データを異なる断面で2次元化したデータ <br><br>左側: 矢状面 <br>中央: 冠状面 <br>右側: 横断面 | ![triple](https://user-images.githubusercontent.com/44970465/107612893-979f0580-6c8a-11eb-9326-2ac426368ec6.png) |


## エラー発生時の問い合わせ
サンプルコードの間違いや動作不具合は本リポジトリのIssuesに投稿ください。

動作不具合についての投稿では、以下を記載ください。

- 実行プログラム名
- エラーメッセージ
- Python、PyTorchなどののライブラリバージョン

## 正誤表
| ページ | 誤 | 正 | 補足 |
|:-----------|:------------|:------------|:------------|
| 52 1行目 | パラメータを待つモデル分布 | パラメータを持つモデル分布| 誤字  |


## 変更履歴
| 日付 | 変更内容 |
|:-----------|:------------|
|2021/02/13　|初版　        |
