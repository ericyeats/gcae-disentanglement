import argparse
import torch

parser = argparse.ArgumentParser(description="Beamsynthesis Training")
parser.add_argument("--devid", help="select a CUDA device for training", default=0)

args = parser.parse_args()

n_dev = torch.cuda.device_count()

device = "cuda:{}".format(int(args.devid) % n_dev) if torch.cuda.is_available() else "cpu"

from disentanglement_utils import FACTOR_VAE, B_TCVAE, beam_s2s2_norm, insert_fake, cmap_colors, gaussian_reconstruction_dist
from disentanglement_utils import DIP_VAE_2
from disentanglement_utils import mig_metric, factor_metric, dci_disentanglement_metric, sap_metric
from architectures import MLP_DEC, MLP_ENC, COND_ARCH, MLP_ENC_VAE, FACTORVAE_COND_ARCH
from gcae_utils import GCAE
from beamsynth_data import BeamSynthData
import numpy as np
import matplotlib.pyplot as plt
from random import randint

loadfunc = lambda path: (beam_s2s2_norm(torch.tensor(np.load(path)).type(torch.float)).squeeze(), float(path[-10:-4]), float(path[-16:-11]),)
target_transform = lambda x: torch.tensor(x)

train_dataset = BeamSynthData('../../../../datasets/bs_data', loadfunc, (".npy",), target_transform=target_transform)

z_dim = 10
use_vae = False
train = True

large_arch = lambda z_dim: FACTORVAE_COND_ARCH(z_dim=z_dim)
small_arch = lambda z_dim: COND_ARCH(z_dim=z_dim, hidden_size=256)
dgae = None

k = info_wgt = kl_wgt = None

if use_vae:
    dgae = FACTOR_VAE(MLP_ENC_VAE(z_dim, (1000,)), MLP_DEC(z_dim, (1000,)), z_dim, large_arch)
    dgae.init_optim_objects(lr_D=0.0001, lr_AE=0.0001, recon_loss=gaussian_reconstruction_dist)
    k=5
    info_wgt = 100.
    kl_wgt = 1.
else:
    dgae = GCAE(MLP_ENC(z_dim, (1000,)), MLP_DEC(z_dim, (1000,)), z_dim, small_arch, uniform_mult=8, gauss_sigma=0.2) # gauss sigma=0.2
    dgae.to(device)
    dgae.init_optim_objects(lr_D=0.0002, lr_AE=0.00005, sample_per_k=1)
    k=5
    info_wgt = 0.0
    kl_wgt = 0.0

dgae.to(device)

colors = cmap_colors(z_dim)


if not train:
    dgae.load_state_dict(torch.load("./models/last_bs.pt"))
    dgae.eval()

if train:

    loss_ds, loss_info, loss_rec, loss_z_kl, mse_recons, var_zs = dgae.fit(train_dataset, 100, batch_per_group=20,\
            batch_size=64, k=k, info_wgt=info_wgt, kl_wgt=kl_wgt, ws_disc=500, rec_ls=False) 

    dgae.eval()

    # save the model
    try:
        torch.save(dgae.state_dict(), "./models/last_bs.pt")
    except PermissionError:
        print("Couldn't open the savefile location")

    first = lambda tup: tup[0]

    # push some examples through the AE
    x = torch.empty(100, 1000)
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
            axs[i][j].plot(x[i*4 + j].squeeze().cpu().numpy())

    plt.savefig('./figs/beamsynthesis/orig.png')

    fig, axs = plt.subplots(4, 4)
    for i in range(4):
        for j in range(4):
            axs[i][j].plot(o[i*4 + j].squeeze().cpu().numpy())

    plt.savefig('./figs/beamsynthesis/recon.png')

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
                        axs[i][j].plot(samples[j].detach().squeeze().cpu().numpy())
                    axs[i][j].get_xaxis().set_visible(False)
                    axs[i][j].get_yaxis().set_visible(False)
        plt.savefig('./figs/beamsynthesis/trav{}.png'.format(k))

    # plot losses

    plt.figure()
    plt.plot(loss_info.abs().log10(), linewidth=2)
    plt.xlabel("Group")
    plt.ylabel("Log10(Info Loss)")
    plt.grid(which='both')

    plt.savefig('./figs/beamsynthesis/info_plot.png')

    # plot log10 reconstruction losses

    group_lst = [i + 1 for i in range(len(loss_rec))]

    plt.figure()
    plt.plot(group_lst, loss_rec.log10(), linewidth=3, label=r'$MSE(\hat{X}, X)$')
    if not use_vae:
        plt.plot(group_lst, mse_recons.log10(), linewidth=3, label=r'$MSE(\tilde{X}, \hat{X})$')
    plt.xlabel("Group")
    plt.ylabel("Log10(Reconstruction Loss)")
    plt.grid(which='both')
    plt.legend()

    plt.savefig('./figs/beamsynthesis/loss_recon_plot.png')

    # plot log10 reconstruction loss with latent variances

    fig, ax1 = plt.subplots()

    ax1.set_label('Group')
    ax1.set_ylabel('Log10(Reconstruction Loss)')
    ax1.plot(group_lst, loss_rec.log10(), linewidth=4, color='k', label=r'$MSE(\hat{X}, X)$')

    ax2 = ax1.twinx() # shares the same x axis

    ax2.set_ylabel('Variance')
    for i in range(dgae.z_dim):
        ax2.plot(group_lst, var_zs[..., i], linewidth=2, label='Var[Z{}]'.format(i), color=colors[i])

    ax2.legend()

    plt.savefig('./figs/beamsynthesis/var_plot.png')

view_alt=5
view_ang=90
alpha=0.5

latent_storage = torch.zeros((len(train_dataset), z_dim))
dgf_storage = torch.zeros((len(train_dataset), 2))

# iterate through the training dataset to create scatter plots of the latent space
with torch.no_grad():
    for i, (data, label) in enumerate(train_dataset):
        data  = data.to(device).unsqueeze(0)
        out = dgae(data)
        z = dgae.z.cpu()
        latent_storage[i] = z
        dgf_storage[i] = label

fig = plt.figure(figsize=(10, 4))
ax = None
ax = fig.add_subplot(121, projection='3d')

for i in range(dgae.z_dim):
    ax.scatter(dgf_storage[..., 1], dgf_storage[..., 0], latent_storage[..., i], label='L{}'.format(i+1), alpha=alpha, color=colors[i])

ax.view_init(view_alt, view_ang)
ax.set_xlabel('S2_duty_cycle')
ax.set_ylabel('S2_frequency')
ax.set_zlabel('Latent Activation')
#ax.legend()

ax = fig.add_subplot(122, projection='3d')

for i in range(dgae.z_dim):
    ax.scatter(dgf_storage[..., 1], dgf_storage[..., 0], latent_storage[..., i], label='L{}'.format(i+1), alpha=alpha, color=colors[i])

ax.view_init(view_alt, view_ang + 45)
ax.set_xlabel('S2_duty_cycle')
ax.set_ylabel('S2_frequency')
ax.set_zlabel('Latent Activation')
ax.legend()

mig = mig_metric(dgae, train_dataset, 2, ((10., 15., 20.), torch.arange(0.2, 0.8, 0.005)[:-1]), 50)
print("MIG: {:1.3f}".format(mig))
fac = factor_metric(dgae, train_dataset, 2, 360, 1000, 200)
print("FAC: {:1.3f}".format(fac))
dci = dci_disentanglement_metric(dgae, train_dataset, 2, 360)
print("DCI: {:1.3f}".format(dci))
sap = sap_metric(dgae, train_dataset, 2, [False, True], n_train = 240, n_test = 120)
print("SAP: {:1.3f}".format(sap))

if train:
        print("Final Info: {:1.3f}\t Final Rec {:1.3f}\t Final KL {:1.3f}".format(loss_info[-1], loss_rec[-1], loss_z_kl[-1]))

