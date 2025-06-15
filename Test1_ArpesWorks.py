import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from tqdm.auto import tqdm
import os                                    #Min ting
import arpesnet as an
import numpy as np



print("Loading trainer")
trainerPath = r"C:\Users\Alexa\Dropbox\Pc\Documents\Privat\Programering\Visual studio\Uni\Bachelor\General coding stuff\ARPES_on_the_fly_arpesnet_206c328\trained_model\arpesnet_n2n_4k.pth"
# trainerPath = r"C:\Users\Alexa\Dropbox\Pc\Documents\Privat\Programering\Visual studio\Uni\Bachelor\Deres\ARPES_on_the_fly_arpesnet_206c328\trained_by_me\arpesnet_60epochs_001.pt"
trainer = an.core.load_trainer(trainerPath)
print("Trainer loaded\n")

encoder = trainer.encoder
decoder = trainer.decoder

encoder.eval()
decoder.eval()


INPUT_SHAPE = (256, 256)
NORM_RANGE = (0, 100)
preprocess = an.transform.Compose(
    [an.transform.Resize(INPUT_SHAPE), an.transform.NormalizeMinMax(*NORM_RANGE)]
)

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "ARPES_on_the_fly_arpesnet_206c328" / "data"
TRAINING_DIR = ROOT_DIR / "train_data" / "train_data"
TEST_DIR = ROOT_DIR / "test_data" / "test_data"
TEST_IMGS_FILE = ROOT_DIR / "test_imgs.pt"
CLUSTER_CENTER_FILE = ROOT_DIR / "cluster_centers.pt"

def getImgFromName(name, range=(0, 0), sum=False):    #Denne fuktion kigger bare igennem de forskellige data- filer og mapper og returnerer de tilhørende billeder.
    imgName = ""
    imgPath = ""
    if (name == "test_imgs.pt"):
        imgName = "test_imgs.pt"
        imgPath = ROOT_DIR
    elif (name == "cluster_centers.pt"):
        imgName = "cluster_centers.pt"
        imgPath = ROOT_DIR
    else:
        paths = [TRAINING_DIR, TEST_DIR]
        for dir in paths:
            if (name in os.listdir(dir)):
                imgName = name
                imgPath = dir
                break
    
    if(imgName == ""):
        print(f'Datafile with name: "{name}" was not found')
        exit()

    print(imgPath / name)
    imgs = torch.load(imgPath / name)

    if(range==(0, 0)): preprocessed = torch.stack([preprocess(img) for img in tqdm(imgs)])
    else:              preprocessed = torch.stack([preprocess(img) for img in tqdm(imgs)][range[0]:range[1]])
    if(sum):
        preprocessed = [torch.sum(preprocessed, dim=0)]


    return name, preprocessed



imgName, test_imgs = getImgFromName("Au(111)_002.pt", sum=False, range=(0, 6))
# print(len(test_imgs))
# exit()


testing_augmentations = an.transform.Compose(
    [
        an.transform.Resize(INPUT_SHAPE),
        an.transform.NormalizeMinMax(*NORM_RANGE),
    ]
)


n_imgs = len(test_imgs)
grid = [
    [f"original_{i}" for i in range(n_imgs)],
    [f"rec_{i}" for i in range(n_imgs)],
    [f"diff_{i}" for i in range(n_imgs)],
]
fig, axes = plt.subplot_mosaic(grid, figsize=(12, 6))
fig.subplots_adjust(hspace=0.375)
fig.suptitle(f"{imgName}", fontsize=16)

test_loss = 0
for i, img in enumerate(test_imgs):
    print(img.size())
    img = testing_augmentations(img)
    rec = decoder(encoder(img.unsqueeze(0)))
    loss = nn.MSELoss()(rec, img).detach().squeeze().cpu().numpy()
    test_loss += loss
    img = img.detach().squeeze().cpu().numpy()
    rec = rec.detach().squeeze().cpu().numpy()
    clim = img.min(), img.max()
    diff = img - rec
    vmax = clim[1]
    axes[f"original_{i}"].imshow(img, cmap="viridis", clim=clim, origin="lower")
    axes[f"rec_{i}"].imshow(rec, cmap="viridis", clim=clim, origin="lower")
    axes[f"diff_{i}"].imshow(diff, cmap="bwr", clim=(-vmax, vmax), origin="lower")
    axes[f"diff_{i}"].set_title(f"MSE: {loss:.3f}")

print(loss/n_imgs)

plt.show()