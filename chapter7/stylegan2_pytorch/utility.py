import numpy as np
import PIL.Image
import math

def adjust_dynamic_range(data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    data = np.rint(data).clip(0, 255).astype(np.uint8)
    return data

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_to_pil_image(image, drange=[-1,1]):
    assert image.ndim == 2 or image.ndim == 3
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0] # grayscale CHW => HW
        else:
            image = image.transpose(1, 2, 0) # CHW -> HWC

    image = adjust_dynamic_range(image, drange, [0,255])
    image = np.rint(image).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if image.ndim == 3 else 'L'
    return PIL.Image.fromarray(image, fmt)

def save_image_grid(images, filename, batch_size, drange):
    grid_width = math.ceil(math.sqrt(batch_size))
    grid_height = math.ceil(batch_size / grid_width)
    grid_size = [grid_width, grid_height]
    pil_img = convert_to_pil_image(create_image_grid(images, grid_size), drange)
    pil_img.save(filename)
