# GANディープラーニング実装ハンドブック

- 第1章：生成モデル    
- 第2章：変分オートエンコーダ(VAE)
- 第3章：GANの基本モデル(DCGAN、CGAN、LSGAN)    
- 第4章：超解像(ESRGAN)    
- 第5章：ドメイン変換(pix2pix、CycleGAN)    
- 第6章：動画変換(Recycle-GAN)
- 第7章：StyleGAN, StyleGAN2
- 第8章：異常検知(AnoGAN、EfficientGAN)
- 第9章：3Dデータの生成(3D-α-WGAN-GP)
- Appendix：理論の補足

## ライブラリのバージョン
- torch:1.7.0
- torchvision:0.8.1
- pandas:1.1.5
- numpy:1.19.5
- matplotlib:3.2.2

## データセット
| 章 | モデル | データセット | ライセンス |取得元リンク|
|:-----------|:------------|:------------|:------------|:------------|
| 2 | AE、VAE | MNIST| Creative Commons  |https://pytorch.org/vision/0.8/datasets.html |
| 3 | DCGAN、CGAN | MNIST| Creative Commons  |https://pytorch.org/vision/0.8/datasets.html |
| 3 | LSGAN、DCGAN | Pet Dataset| Creative Commons  |https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| 4 | ESRGAN | Pet Dataset| Creative Commons  |https://www.robots.ox.ac.uk/~vgg/data/pets/ |
| 5 | pix2pix、CycleGAN | facades | Creative Commons  |https://cmp.felk.cvut.cz/~tylecr1/facade/ |
| 6 | Cycle GAN、Recycle-GAN | VidTIMIT Audio-Video Dataset| リンク先のLICENSEに利用時の注事事項の記載あり|https://conradsanderson.id.au/vidtimit/ |
| 7 | StyleGAN, StyleGAN2 | Endless Summer Dataset| データセット内のファイルを画像として閲覧できる方法でinternet等で公開したり、販売するのは禁止です。 |https://drive.google.com/file/d/1LM4FtUltzS45PuFyfuSp3I8QdTD8Cu0F/view?usp=sharing |
| 8 | AnoGAN、EfficientGAN | Fruits 360 dataset | Creative Commons  |https://github.com/antonnifo/fruits-360 |
| 9 | 3D-α-WGAN-GP | IXI Dataset| Creative Commons |http://brain-development.org/ixi-dataset/ |


## 学習時の注意点
| 章 | モデル | 注意点 |学習の目安時間 |
|:-----------|:------------|:------------|:------------|
| 3 | LSGAN、DCGAN | LSGANよりDCGANの方が猫っぽい画像を生成します。| 5時間程度|
| 4 | ESRGAN | デフォルト設定だとファイルはColabに保存されるので、Google Driveに保存したい場合はセル[36]で保存先をGoogle Driveに変更してください。| 5～6時間程|
| 5 | pix2pix, CycleGAN | 特になし| ５〜８時間程度 |
| 6 |CycleGAN 、Recycle-GAN| 特になし| 半日から数日 |
| 7 | StyleGAN、StyleGAN2 | GPUは執筆時点で最速のP100を推奨（理想はV100）。V100は Colab Proで利用可能ですが、日本では執筆時点で利用はできません。| P100で2週間程度|
| 8 | xx | xx| xx |
| 9 | 3D-α-WGAN-GP | 特になし | 8時間程度|


## 生成画像の例
| 章 | モデル |説明| 生成画像 |
|:--|:---------|:------------|:------------------------------------------|
| 3 | DCGAN |ノイズからサイズ128×128の猫画像を生成 | <img width="256" alt="fake_cat" src="https://user-images.githubusercontent.com/40778791/107382141-e8034f80-6b32-11eb-8029-563864814cfd.png"> |
| 4 | xx | xx |xx |xx|
| 5 | CycleGAN |xx | ![0089](https://user-images.githubusercontent.com/20309500/107369273-eb8fda00-6b24-11eb-9e5b-623f1666403f.png) |
| 6 | Recycle-GAN |xx | ![A2B](https://user-images.githubusercontent.com/15444879/107294483-d1b4af80-6ab0-11eb-82b7-a44c322eb403.png) |
| 7 | StyleGAN2 |xx|![2021 02 08_01](https://user-images.githubusercontent.com/21982866/107151130-52d25080-69a4-11eb-83ff-1d6d24c642cc.png)![2021 02 08_05](https://user-images.githubuserconxx |tent.com/21982866/107151201-a80e6200-69a4-11eb-9eaa-d2c6d485787d.png)![2021 02 08_09](https://user-images.githubusercontent.com/21982866/107151263-f7549280-69a4-11eb-8d2f-7ccdefaab869.png)|
| 8 | xx |xx | xx |
| 9 | 3D-α-WGAN-GP |xx | ![triple](https://user-images.githubusercontent.com/44970465/107247215-6351fc00-6a74-11eb-8c37-7de5e8457846.png) |


## 正誤表
| ページ | 誤 | 正 | 補足 |
|:-----------|:------------|:------------|:------------|
| xx | xx | xx| xx  |


## 変更履歴
| 日付 | 内容 |
|:-----------|:------------|
|2/13　|初版　|


