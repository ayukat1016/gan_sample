# GANディープラーニング実装ハンドブック

- 第1章：生成モデル    
- 第2章：変分オートエンコーダ(VAE)
- 第3章：GANの基本モデル(DCGAN、CGAN、LSGAN)    
- 第4章：超解像(ESRGAN)    
- 第5章：ドメイン変換(pix2pix、CycleGAN)    
- 第6章：動画変換(Recycle-GAN)
- 第7章：StyleGAN
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
| 5 | pix2pix、CycleGAN | xx| xx  |xx |
| 6 | Cycle GAN、Recycle-GAN | VidTIMIT Audio-Video Dataset| リンク先のLICENSEに利用時の注事事項の記載あり  |https://conradsanderson.id.au/vidtimit/ |
| 7 | Style GAN | Endless Summer Datasets| xx  |xx |
| 8 | AnoGAN、EfficientGAN | Fruits 360 dataset | Creative Commons  |https://github.com/antonnifo/fruits-360 |
| 9 | 3D-α-WGAN-GP | IXI Datasets| Creative Commons |http://brain-development.org/ixi-dataset/ |


## 学習時の注意点
| 章 | モデル | 注意点 |学習の目安時間 |
|:-----------|:------------|:------------|:------------|
| 3 | LSGAN、DCGAN | 1100エポック前後の猫画像が性能良い| 猫が生成されるまで4時間程度の学習が必要|
| 4 | ESRGAN | デフォルト設定だとファイルはColabに保存されるので、Google Driveに保存したい場合はセル[36]で保存先をGoogle Driveに変更してください。| 5～6時間程|
| 7 | StyleGAN | GPUは執筆時点で最速のP100を推奨（理想はV100）。V100は Colab Proで利用可能ですが、日本では執筆時点で利用はできません。| P100で2週間程度|


## 正誤表
| ページ | 誤 | 正 | 補足 |
|:-----------|:------------|:------------|:------------|
| xx | xx | xx| xx  |


## 変更履歴
| 日付 | 内容 |
|:-----------|:------------|
|2/13　|初版　|


