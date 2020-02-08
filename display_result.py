import os
import numpy as np
import torch
import torchvision.utils as vutils
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

dir_result = './results/stn/mnist/images'
lst_result = os.listdir(dir_result)

name_input = [f for f in lst_result if f.endswith('input.png')]
name_input_stn = [f for f in lst_result if f.endswith('input_stn.png')]

name_input.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
name_input_stn.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

dataset_test = datasets.MNIST(root='.', train=False, download=True)
loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

##
labels = loader_test.dataset.targets.numpy()

##
nx = 28
ny = 28
nch = 1

n = 10
m = 10

input = torch.zeros((n*m, ny, nx, nch))
input_stn = torch.zeros((n*m, ny, nx, nch))

for i in range(n):
    idx = np.where(labels == i)[0]
    np.random.shuffle(idx)
    for j in range(m):
        k = m * i + j
        input[k, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, name_input[idx[j]]))[:, :, :nch])
        input_stn[k, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, name_input_stn[idx[j]]))[:, :, :nch])

input = input.permute((0, 3, 1, 2))
input_stn = input_stn.permute((0, 3, 1, 2))

plt.figure(figsize=(n, m))

plt.subplot(121)
plt.imshow(np.transpose(vutils.make_grid(input, nrow=n, padding=2, normalize=True), (1, 2, 0)))
plt.axis('off')
plt.axis('image')
plt.title('MNIST Input')

plt.subplot(122)
plt.imshow(np.transpose(vutils.make_grid(input_stn, nrow=n, padding=2, normalize=True), (1, 2, 0)))
plt.axis('off')
plt.axis('image')
plt.title('STN Output')

plt.show()

