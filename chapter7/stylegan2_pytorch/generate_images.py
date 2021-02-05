import argparse
import os
import glob
import re
import numpy as np
from datetime import datetime
import torch
from utility import convert_to_pil_image
from base_layer import BaseLayer

def _parse_num_range(s):
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def main(opt):
    _, generatro_predict, _ = BaseLayer.restore_model(opt.model_path)
    generatro_predict.truncation_psi = opt.truncation_psi

    if not os.path.isdir(opt.output_path):
        os.makedirs(opt.output_path, exist_ok=True)

    for index, seed in enumerate(opt.seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, opt.latent_dim)
        z = Tensor(np.tile(z, (opt.batch_size, 1)))
        images, _ = generatro_predict(z)
        images = images.to('cpu').detach().numpy()
        pil_img = convert_to_pil_image(images[0])
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(opt.output_path, '{}_{}.png'.format(now, seed))
        pil_img.save(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./model', help="model_path")
    parser.add_argument("--output_path", type=str, default='./results', help="output path of generate_images")
    parser.add_argument("--seeds", type=_parse_num_range, default='1000-1005', help="seeds")
    parser.add_argument("--truncation-psi", type=float, default='0.5', help="truncation_psi")
    parser.add_argument("--latent_dim", type=int, default=512, help="")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--resolution", type=int, default=128, help="size of each image dimension")
    option = parser.parse_args()
    print(option)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    main(option)
