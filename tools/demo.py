
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

import time

torch.set_grad_enabled(False)
np.random.seed(123)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png',)
args = parse.parse_args()
cfg = cfg_factory[args.model]


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.to(device)

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
print("demo image shape: {}".format(im.shape))
im = cv2.resize(im, (640, 480))
print("imput image shape: {}".format(im.shape))
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).to(device)

# inference
inference_num = 10
for i in range(inference_num):
    start_time = time.time()
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    end_time = time.time()
    print("inference_time:{}".format(end_time - start_time))
pred = palette[out]
cv2.imwrite('./res.jpg', pred)
