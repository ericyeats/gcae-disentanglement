import argparse
import torch

parser = argparse.ArgumentParser(description="dSprites Training")
parser.add_argument("--devid", help="select a CUDA device for training", default=0)

args = parser.parse_args()

n_dev = torch.cuda.device_count()

device = "cuda:{}".format(int(args.devid) % n_dev) if torch.cuda.is_available() else "cpu"

from disentanglement_utils import FACTOR_VAE, B_TCVAE, DIP_VAE_2, dsprites_norm, insert_fake, cmap_colors, bernoulli_reconstruction_dist
from disentanglement_utils import mig_metric, factor_metric, dci_disentanglement_metric, sap_metric
from architectures import COND_ARCH, MLP_ENC, MLP_ENC_VAE, MLP_DEC, FACTORVAE_COND_ARCH
from gcae_utils import GCAE
from dsprites_data import DSPRITES
from torchvision.transforms import Compose


import matplotlib.pyplot as plt
from random import randint




def dsprites_reshape(x):
    x = torch.tensor(x, dtype=torch.float)
    if x.dim() == 2:
        x = x.view(1, 64, 64)
    else: # batch version
        x = x.view(x.shape[0], 1, 64, 64)
    return x

gauss_sigma = 0.2
z_dim = 10
use_vae = False
train = True
batch_size=256

tform = dsprites_reshape if use_vae else Compose([dsprites_reshape, dsprites_norm])
train_dataset = DSPRITES(transform=tform)

large_arch = lambda z_dim: FACTORVAE_COND_ARCH(z_dim=z_dim)
small_arch = lambda z_dim: COND_ARCH(z_dim=z_dim, hidden_size=256)
dgae = None

modelname = "none"
kl_save = None

if use_vae:
    dgae = FACTOR_VAE(MLP_ENC_VAE(z_dim, (1, 64, 64)), \
        MLP_DEC(z_dim, (1, 64, 64)), z_dim, large_arch, out_act=torch.nn.Sigmoid())
    dgae.init_optim_objects(lr_D=0.0001, lr_AE=0.0001, recon_loss=bernoulli_reconstruction_dist)
    k=1
    info_wgt = 0.
    kl_wgt = 1.

    modelname = "FACV"
    if isinstance(dgae, B_TCVAE): 
        modelname = "BTCV"
    elif isinstance(dgae, DIP_VAE_2):
        modelname = "DIPV"
    kl_save = kl_wgt
else:
    dgae = GCAE(MLP_ENC(z_dim, (1, 64, 64)), MLP_DEC(z_dim, (1, 64, 64)), z_dim, small_arch, uniform_mult=8, gauss_sigma=gauss_sigma) # gauss sigma=0.2
    dgae.init_optim_objects(lr_D=0.0002, lr_AE=0.00005, sample_per_k=1)
    k=5
    info_wgt = 0.2
    kl_wgt = 0.0

    modelname = "GCAE"
    kl_save = gauss_sigma

savefile = "./models/dsprites/{}_id{}_k{}_iw{:1.2f}_ks{:1.2f}_model.pt".format(modelname, args.devid, k, info_wgt, kl_save)

print("will save to: ", savefile)

dgae.to(device)

# for plotting
colors = cmap_colors(z_dim)

if not train:
    dgae.load_state_dict(torch.load(savefile))

    dgae.eval()

if train:

    loss_ds, loss_info, loss_rec, loss_z_kl, mse_recons, var_zs = dgae.fit(train_dataset, 1000, batch_per_group=20,\
            batch_size=batch_size, k=k, info_wgt=info_wgt, kl_wgt=kl_wgt, ws_disc=500) 

    dgae.eval()

    # save the model
    try:
        torch.save(dgae.state_dict(), savefile)
    except PermissionError:
        print("Couldn't open the savefile location")

    # plot losses

    plt.figure()
    plt.plot(loss_info.abs().log10(), linewidth=2)
    plt.xlabel("Group")
    plt.ylabel("Log10(Info Loss)")
    plt.grid(which='both')

    plt.savefig('./figs/dsprites/info_plot.png')

    # plot log10 reconstruction loss

    group_lst = [i + 1 for i in range(len(loss_rec))]

    plt.figure()
    plt.plot(group_lst, loss_rec.log10(), linewidth=2)
    plt.xlabel("Group")
    plt.ylabel("Log10(Reconstruction Loss)")
    plt.grid(which='both')

    # plot log10 reconstruction loss with latent variances

    fig, ax1 = plt.subplots()

    ax1.set_label('Group')
    ax1.set_ylabel('Log10(Reconstruction Loss')
    ax1.plot(group_lst, loss_rec.log10(), linewidth=4, color='k')

    ax2 = ax1.twinx() # shares the same x axis

    ax2.set_ylabel('Variance')
    for i in range(dgae.z_dim):
        ax2.plot(group_lst, var_zs[..., i], linewidth=2, label='Z{}'.format(i), color=colors[i])

    ax2.legend()

    plt.savefig('./figs/dsprites/var_plot.png')

dgae.eval()

first = lambda tup: tup[0]

# push some examples through the AE
x = torch.empty(100, 1, 64, 64)
for i in range(100):
    ind = randint(0, len(train_dataset)-1)
    x[i] = first(train_dataset[ind])

x = x.to(device)
o = dgae(x).detach()
z = dgae.z.detach()
z_max = z.max(dim=0)[0]
z_min = z.min(dim=0)[0]
coverages = z_max - z_min
print("Maxes: ", z_max)
print("Mins: ", z_min)
print("Coverages: ", coverages)

fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i][j].imshow(x[i*4 + j].squeeze().cpu().numpy(), cmap='gray')

plt.savefig('./figs/dsprites/originals.png')

fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i][j].imshow(o[i*4 + j].squeeze().cpu().numpy(), cmap='gray')

plt.savefig('./figs/dsprites/reconstructions.png')

# perform latent traversals for a few images
z = dgae.z.detach()
for k in range(3):
    fix, axs = plt.subplots(z_dim, 10)
    for i in range(z_dim):
            _z = z[k].detach().unsqueeze(0).tile((10, 1))
            dec_in = insert_fake(_z, torch.linspace(z_min[i], z_max[i], 10), i)
            samples = dgae.dec(dec_in)
            for j in range(10):
                if coverages[i] > 0.2:
                    axs[i][j].imshow(samples[j].detach().squeeze().cpu().numpy(), cmap='gray')
                axs[i][j].get_xaxis().set_visible(False)
                axs[i][j].get_yaxis().set_visible(False)

    plt.savefig('./figs/dsprites/trav{}.png'.format(k+1))

# get random sample of 10000 datapoints for MIG
dspr_sample = train_dataset.dsprites_sample(10000)
# should already have transform applied
assert len(dspr_sample) == 10000
mig = mig_metric(dgae, dspr_sample, 6, train_dataset.possible_lat_values, 20)
print("MIG: {:1.3f}".format(mig))
fac = factor_metric(dgae, train_dataset, 6, 10000, 5000, 1000)
print("FAC: {:1.3f}".format(fac))
dci = dci_disentanglement_metric(dgae, train_dataset, 6, 1000)
print("DCI: {:1.3f}".format(dci))
sap = sap_metric(dgae, train_dataset, 6, [False, False, True, True, True, True], n_train = 5000, n_test = 1000)
print("SAP: {:1.3f}".format(sap))

if train:
    print("Final Info: {:1.3f}\t Final Rec {:1.3f}\t Final KL {:1.3f}".format(loss_info[-1], loss_rec[-1], loss_z_kl[-1]))

plt.show()