from torchvision.datasets import DatasetFolder
import torch

class BeamSynthData(DatasetFolder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latents_sizes = torch.tensor([3, 120], dtype=torch.long)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple (sample, target) where target is a tuple of DGF information for the sample
        """
        if torch.is_tensor(index) and index.numel() > 1:
            index = index.tolist()
            seq = [self.__getitem__(ind) for ind in index]
            seq_data = [s[0].unsqueeze(0) for s in seq]
            seq_lab = [s[1].unsqueeze(0) for s in seq]
            data, lab =  torch.cat(seq_data, dim=0), torch.cat(seq_lab, dim=0)
            #print(type(data))
            return data, lab
        
        path, target = self.samples[index]
        sample_tup = self.loader(path)
        sample = sample_tup[0]
        target = sample_tup[1:]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    @staticmethod
    def latent_to_index(factors):
        # factors is a torch array of (batch_size, fac indices)
        bases = torch.tensor([[120, 1]], dtype=torch.long).t()
        return (factors @ bases).squeeze_() # should be (batch_size)

    def sample_latent(self, count=1):
        samples = torch.zeros((count, self.latents_sizes.numel()), dtype=torch.long)
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = torch.randint(lat_size, (count,))
        return samples

    def sample_latent_factor_metric(self, size=1):
        # choose a latent to hold constant
        const_lat = torch.randint(2, (1,))
        # randomly sample a batch of latents
        samples = self.sample_latent(size)
        # iterate through all the samples and sample a new latent value for the constant lat
        lat_val_ind = torch.randint(self.latents_sizes[const_lat].item(), (1,)).item()
        for i in range(size):
            samples[i][const_lat] = lat_val_ind
        # index into the dataset and return both batches
        indices = self.__class__.latent_to_index(samples)

        data, _ = self.__getitem__(indices)
        return const_lat, data