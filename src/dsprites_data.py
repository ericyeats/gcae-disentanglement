from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
# Load dataset
_path = '../../../../datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
_path_log_probs = '../../../../datasets/dsprites/dsprites_logprobs.npy'


def prod(shape):
    a = 1
    for s in shape:
        a *= s
    return a

class DSPRITES(torch.utils.data.Dataset):
    
    def __init__(self, path=_path, transform=None, omit_rotation=False, include_log_prob=False, log_prob_path=_path_log_probs):
        super(DSPRITES, self).__init__()
        # use naming conventions from https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
        dataset_zip = np.load(path)
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_sizes = np.array([ 1,  3,  6, 40, 32, 32])
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
        self.transform = transform
        self.omit_rotation = omit_rotation
        self.include_log_prob = include_log_prob and omit_rotation
        if self.include_log_prob:
            self.log_prob = np.load(log_prob_path)
        self.possible_lat_values = [
            np.arange(1),
            np.arange(3)*(2.-0.)/2.+0., # (maxval-minval)/(n_val-1)+minval
            np.arange(6)*(1.-0.5)/5.+0.5,
            np.arange(40)*(2.*np.pi-0.)/39.+0.,
            np.arange(32)*(1.-0.)/31.+0.,
            np.arange(32)*(1.-0.)/31.+0.
        ]
        if self.omit_rotation:
            self.possible_lat_values[3] = np.arange(1)
        
    def __len__(self):
        return self.imgs.shape[0] if not self.omit_rotation else self.imgs.shape[0] // 40

    def adjusted_index(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # if omit_rotation, need to calculate the adjusted idx based on the original idx
        if self.omit_rotation:
            to_list = False
            if not isinstance(idx, list):
                idx = [idx]
                to_list = True
            y_pos = [i % 32 for i in idx]
            x_pos = [(i // 32) % 32 for i in idx]
            scale = [(i // (32*32)) % 6 for i in idx]
            shape = [(i // (32*32*6)) % 3 for i in idx]
            n = len(y_pos)
            idx = [self.latent_to_index(np.array(z)) for z in zip(np.zeros(n), shape, scale, np.zeros(n), x_pos, y_pos)]
            if to_list:
                idx = idx[0]
        return idx
    
    def __getitem__(self, idx):
        idx = self.adjusted_index(idx)
        latents = torch.tensor(self.latents_values[idx])
        if self.include_log_prob:
            log_prob = torch.tensor(self.log_prob[idx])
            if latents.dim() == 1:
                latents = torch.cat((latents, log_prob.view(1)), dim=0)
            else:
                latents = torch.cat((latents, log_prob.view(-1, 1)), dim=1)
        data = self.imgs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data, latents

    def __iter__(self):
        self.__iter_idx = 0
        return self
    
    def __next__(self):
        if self.__iter_idx >= len(self):
            raise StopIteration("reached end of the dSprites dataset")
        item = self.__getitem__(self.__iter_idx)
        self.__iter_idx += 1
        return item

    def dsprites_sample(self, size):
        idx = torch.randperm(len(self))[:size]
        data_ten, lab_ten = self.__getitem__(idx)
        return DSPRITES_SAMPLE(data_ten, lab_ten)
    
    # expects adjusted index
    def latent_to_index(self, latents):
        return (latents @ self.latents_bases).astype(int)

    def index_to_latent(self, index):
        index = self.adjusted_index(index)
        return [(index // prod(self.latents_sizes[v+1:]))%self.latents_sizes[v] for v in range(6)]

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)
        if self.omit_rotation:
            samples[:, 3] = 0 # enforce zero rotation
        return samples
    
    # TODO - make compatible with omit_rotation
    def sample_latent_dm(self, size=1):
        # choose a latent to hold constant
        const_lat = np.random.randint(2, 6)
        # randomly sample two batches of latents
        samples1 = self.sample_latent(size)
        samples2 = self.sample_latent(size)
        # iterate through all the samples and sample a new latent value for the constant lat
        for i in range(size):
            lat_val_ind = np.random.randint(0, self.latents_sizes[const_lat])
            samples1[i][const_lat] = lat_val_ind
            samples2[i][const_lat] = lat_val_ind
        # index into the dataset and return both batches
        data1 = self.__getitem__(self.latent_to_index(samples1))
        data2 = self.__getitem__(self.latent_to_index(samples2))
        return const_lat, data1, data2

    def sample_latent_factor_metric(self, size=1):
        # choose a latent to hold constant
        const_lat = np.random.randint(1, 6)
        # randomly sample a batch of latents
        samples = self.sample_latent(size)
        # iterate through all the samples and sample a new latent value for the constant lat
        lat_val_ind = np.random.randint(0, self.latents_sizes[const_lat])
        for i in range(size):
            samples[i][const_lat] = lat_val_ind
        # index into the dataset and return both batches
        data = self.__getitem__(self.latent_to_index(samples))[0]
        return const_lat, data


class DSPRITES_SAMPLE(torch.utils.data.Dataset):

    def __init__(self, data_ten, lab_ten, tform=None):
        self.data_ten = data_ten
        self.lab_ten = lab_ten
        self.tform = tform

    def __len__(self):
        return self.data_ten.shape[0]

    def __getitem__(self, idx):
        data, lab = self.data_ten[idx], self.lab_ten[idx]
        if self.tform is not None:
            data = self.tform(data)
        return data, lab


def test():

    with torch.no_grad():

        dataset = DSPRITES(omit_rotation=True, include_log_prob=True)

        import matplotlib.pyplot as plt

        # plot 4 images
        fig, axs = plt.subplots(2, 2)
        for i in range(4):
            dat, lat = dataset[i]
            axs[i//2][i%2].imshow(dat.squeeze())
            axs[i//2][i%2].set_title(lat)
        
        plt.show()

