import glob
from PIL import Image
import torch.utils.data as data


class TrainDataset(data.Dataset):
    def __init__(self, file_path, transform=None, phase='train'):
        self.file_list = self.make_datapath_list(file_path)
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # index番目の画像をロード
        img_path = self.file_list[index]
        img = Image.open(img_path)  # [高さ][幅][色RGB]

        # 画像の前処理を実施
        img_transformed = self.transform(img)  # torch.Size([3, 224, 224])

        return img_transformed

    def make_datapath_list(self, file_path):
        path_list = []

        # globを利用してサブディレクトリまでファイルパスを取得する
        for path in glob.glob(file_path):
            path_list.append(path)
        return path_list
