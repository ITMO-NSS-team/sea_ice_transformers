import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
from datetime import datetime


# Script to visualize prediction

name = "104_52_results"
path_to_train = rf"D:\Projects\test_cond\AAAI_code\Ice/kara"
path_to_dir = r"D:\Projects\SwinLSTM\npy_results_104_52_7days_kara\kara"
dir_list = os.listdir(path_to_dir)
check_in_data = ["osi_" + i.split("_")[1] for i in dir_list]
data_gt = [path_to_train + f"/{i}" for i in check_in_data]
dates = [datetime.strptime(i.split("_")[1].split(".")[0], "%Y%m%d") for i in dir_list]
fig = plt.figure(figsize=(12, 18))

columns = 4
rows = 4
step = 3
start = 52
range_ = 100
n = 0
for i in range(start, range_, step):
    img = resize(
        np.load(path_to_dir + "/" + dir_list[i]), (140, 140), anti_aliasing=False
    )
    img = np.fliplr(img)
    img = np.rot90(img, 2)
    fig.add_subplot(rows, columns, n + 1)
    n += 1
    plt.axis("off")
    plt.imshow(img, cmap="Blues_r", aspect="auto")
    plt.title(f"{dates[i].year}-{dates[i].month}-{dates[i].day}", fontsize=21)
plt.savefig(f"images/{range_}_predict_{name}_step_{step}.png", dpi=300)
plt.show()

fig = plt.figure(figsize=(12, 18))

n = 0
for i in range(start, range_, step):
    img = resize(np.load(data_gt[i]), (140, 140), anti_aliasing=False)
    img = np.fliplr(img)
    img = np.rot90(img, 2)
    fig.add_subplot(rows, columns, n + 1)
    n += 1
    plt.axis("off")
    plt.imshow(img, cmap="Blues_r", aspect="auto")
    plt.title(f"{dates[i].year}-{dates[i].month}-{dates[i].day}", fontsize=21)
plt.savefig(f"images/{range_}_GT_{name}_step_{step}.png", dpi=300)
plt.show()
