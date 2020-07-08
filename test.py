import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import torchvision
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

width = 512
height = 512
transforms_ = [
    transforms.Resize((height,width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_dataloader = DataLoader(
    ImageDataset(".", transforms_=transforms_, mode="val2"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


transform = transforms.Compose([
    transforms.Resize((width,height), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

generator = GeneratorUNet()

if cuda:
    generator = generator.cuda()

generator.load_state_dict(torch.load("generator_195.pth"))
img = cv2.imread('79.png',1)
for i, batch in enumerate(val_dataloader):
	real_A = Variable(batch["B"].type(Tensor))
	real_B = Variable(batch["A"].type(Tensor))
	fake_B = generator(real_A)
	img_sample = torch.cat((real_A.data, fake_B.data), -2)
	save_image(img_sample, "test"+str(i)+".png" ,normalize=True)
