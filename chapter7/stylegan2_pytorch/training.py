import argparse
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

from tensorboard_logger import TensorboardLogger
from trainer import Trainer

def main(opt):
    # tensorboardに出力するためのwriterを作成
    TensorboardLogger.writer = SummaryWriter(log_dir=opt.tensorboard_path)

    # Trainerクラスのインスタンスを作成
    trainer = Trainer(opt)

    for current_loop_num in range(opt.max_loop_num):
        # generatorの学習
        g_loss = trainer.train_generator(current_loop_num)

        # discriminatorの学習
        d_loss = trainer.train_discriminator(current_loop_num)

        # ログの出力
        print('current_loop_num: {}, d_loss: {}, g_loss: {}'.format(
            current_loop_num,
            d_loss,
            g_loss))

        if current_loop_num % opt.save_model_interval == 0:
            # modelを保存する
            trainer.save_model()

        if current_loop_num % opt.fid_score_interval == 0 and 0 < current_loop_num:
            # FID scoreを計算する
            trainer.calculate_fid_score()

        TensorboardLogger.global_step += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_loop_num", type=int, default=65536, help="メインループの回数を設定します")
    parser.add_argument("--data_path", type=str, default='../dataset/endless_summer', help="学習に使用するデータセットのディレクトリを指定します")
    parser.add_argument("--batch_size", type=int, default=16, help="ミニバッチのサイズを指定します")
    parser.add_argument("--latent_dim", type=int, default=512, help="潜在変数の次元数を指定します")
    parser.add_argument("--learning_rate", type=float, default=0.002, help="adamの学習率を指定します")
    parser.add_argument("--beta1", type=float, default=0.0, help="adamの減衰率1を指定します")
    parser.add_argument("--beta2", type=float, default=0.99, help="adamの減衰率2を指定します")
    parser.add_argument("--resolution", type=int, default=32, help="生成する画像の解像度を指定します")
    parser.add_argument("--g_reg_interval", type=int, default=4, help="generatorの正則化処理を何回に一回行うかを指定します。")
    parser.add_argument("--d_reg_interval", type=int, default=16, help="discriminatorの正則化処理を難解に一回行うかを指定します")
    parser.add_argument("--model_path", type=str, default='./model', help="モデルを保存するディレクトリを指定します")
    parser.add_argument("--results", type=str, default='./results', help="学習途中に生成した画像を保存するディレクトリを指定します")
    parser.add_argument("--is_restore_model", type=strtobool, default=True, help="保存したモデルファイルを読み込んで学習を始める場合はTrueを設定します")
    parser.add_argument("--cache_path", type=str, default='./cache', help="計算で使う作業ファイルを置く場所を指定します")
    parser.add_argument("--tensorboard_path", type=str, default='./logs', help="Tensorboardのファイルを格納するディレクトリを指定します")
    parser.add_argument("--save_model_interval", type=int, default=128, help="モデルを保存する頻度を指定します")
    parser.add_argument("--fid_score_interval", type=int, default=3072, help="fidを計算する頻度を指定します")
    parser.add_argument("--save_metrics_interval", type=int, default=4, help="メトリックスをTensorboardに保存する頻度を指定します")
    parser.add_argument("--save_images_tensorboard_interval", type=int, default=32, help="生成画像をTensorboardに保存する頻度を指定します")
    parser.add_argument("--save_images_interval", type=int, default=128, help="生成画像を保存する頻度を指定します")
    parser.add_argument("--generator_train_num", type=int, default=4, help="generatorが連続で学習する回数を指定します")
    parser.add_argument("--discriminator_train_num", type=int, default=4, help="discriminatorが連続で学習する回数を指定します")
    parser.add_argument("--adjust_decay_param", type=float, default=1.0, help="推論用のgeneratorに重みを適用するパラメータを調整します。")
    parser.add_argument("--reverse_decay", type=float, default=1.0, help="")
    option = parser.parse_args()
    print(option)

    main(option)

