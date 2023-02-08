import torch
from disentanglement_utils import create_latent_image, tile_fake, information_functional, bnd
from beamsynth_data import BeamSynthData
import time

class LatActTanh(torch.nn.Module):

    def __init__(self, scl=1.0):
        super().__init__()
        self.scl = scl

    def forward(self, x):
        return self.scl*torch.tanh(x/self.scl)

class LatActSoftsign(torch.nn.Softsign):

    def __init__(self, scl=1.0):
        super().__init__()
        self.scl = scl

    def forward(self, x):
        return self.scl*super().forward(x/self.scl)



class GCAE(torch.nn.Module):

    def __init__(self, enc, dec, z_dim, disc_arch, uniform_mult=1, gauss_sigma=0.1, lat_bnd=3.):
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.z_dim = z_dim
        self.discs = torch.nn.ModuleList([disc_arch(z_dim) for _ in range(self.z_dim)])
        self.uniform_mult = uniform_mult
        self.gauss_sigma = gauss_sigma
        
        self.lat_bnd = lat_bnd
        self.lat_act = LatActSoftsign(lat_bnd)

    def forward(self, x):
        self.z_phi_pre = self.enc(x)
        self.z_phi = self.lat_act(self.z_phi_pre)
        # apply noise if training, do not do so during evaluation
        if self.training:
            self.z = self.z_phi + self.gauss_sigma*torch.randn_like(self.z_phi)
            self.z_psi = self.z #+ self.gauss_sigma*torch.randn_like(self.z)
        else:
            self.z = self.z_phi.clone()
            self.z_psi = self.z.clone()
        self.x_hat = self.dec(self.z_psi)
        return self.x_hat
    
    def init_optim_objects(self, lr_D=0.001, lr_AE=0.001, sample_per_k=1):
        self.AE_optim = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr_AE, weight_decay=0)
        self.Ds_optim = torch.optim.Adam(self.discs.parameters(), lr=lr_D, betas=(0.5, 0.9), weight_decay=0)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.recon_loss = torch.nn.MSELoss()
        self.sample_per_k = sample_per_k

    def to(self, device):
        self = super().to(device)
        self.device = device
        return self

    def cpu(self):
        return self.to("cpu")
    
    def real_sample(self, z):
        return z

    def fake_sample(self, z, ind=None):
        if ind is None:
            return z.clone().uniform_(-bnd, bnd) # want this in denominator of KL
        else:
            z_clone = z.clone()
            z_clone[..., ind] = torch.empty_like(z_clone[..., ind]).uniform_(-bnd, bnd)
            return z_clone

    def train_discriminator(self, dataloader, loss_Ds_agg, dz_agg, dgz_agg):
        # train the discriminator on real data
        self.Ds_optim.zero_grad()
        # define the joint data distribution
        data, label = next(dataloader)
        data = data.to(self.device)
        batch_size = data.shape[0]
        out = self.forward(data) # generate z encodings. defines the 'true' distribution 
        z = self.z.detach()

        z_real = self.real_sample(z)

        pred_real_d = [d(z_real) for d in self.discs]
        l_discs = sum([self.criterion(p_r_d, torch.ones_like(p_r_d)).sum() for p_r_d in pred_real_d])

        z_fake_d = [self.fake_sample(z.tile(self.uniform_mult, 1), i).detach() for i in range(self.z_dim)]
        pred_fake_d = [d(z_f_d) for d, z_f_d in zip(self.discs, z_fake_d)]
        l_discs += sum([self.criterion(p_f_d, torch.zeros_like(p_f_d)).sum() for p_f_d in pred_fake_d])

        # Step the discriminator
        l_discs /= batch_size*(1. + self.uniform_mult)

        (l_discs).backward()
        self.Ds_optim.step()

        loss_Ds_agg += l_discs.item()

        return loss_Ds_agg, dz_agg, dgz_agg

    def fit(self, dataset, n_group, batch_per_group=1000, batch_size=64, k=1, info_wgt=0.0, kl_wgt=1.0, ws_disc=0, rec_ls=False):
        start_time = time.time()
        assert info_wgt >= 0. and kl_wgt >= 0.
        self.train()

        # set up loss storages
        loss_Ds = torch.zeros(n_group)
        loss_AE_info = torch.zeros(n_group)
        loss_AE_rec = torch.zeros(n_group)
        loss_Z_kl = torch.zeros(n_group)
        mse_recons = torch.zeros(n_group)
        var_zs = torch.zeros(n_group, self.z_dim)

        # define samplers for the Discriminator
        sample_per_batch = self.sample_per_k*k + 1
        n_samples = batch_size*batch_per_group*n_group*sample_per_batch + ws_disc*batch_size*self.sample_per_k
        random_sampler = torch.utils.data.RandomSampler(dataset, \
                    replacement=True, num_samples=n_samples)
        batch_sampler = torch.utils.data.BatchSampler(random_sampler, batch_size=batch_size, drop_last=False)
        dataloader = iter(torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler))

        if isinstance(dataset, BeamSynthData) and rec_ls:
            create_latent_image(self, dataset, 0) # initial latent space

        for _ in range(ws_disc):
            _, _, _ = self.train_discriminator(dataloader, 0., 0., 0.)

        for g in range(n_group):
            loss_Ds_agg = 0.
            dz_agg = 0.
            dgz_agg = 0.
            l_AE_info_agg = 0.
            l_AE_rec_agg = 0.
            l_Z_kl_agg = 0.
            mse_recons_agg = 0.
            var_zs_agg = torch.zeros(self.z_dim).to(self.device)

            for b in range(batch_per_group):
                print("\rGroup: {}\t{:2.0f}%".format(g+1, 100*(b+1)/batch_per_group), end="")

                # begin by training the Discriminators/Generators for K steps
                for _ in range(k):

                    loss_Ds_agg, dz_agg, dgz_agg = self.train_discriminator(dataloader, loss_Ds_agg, dz_agg, dgz_agg)

                # train the AE for reconstruction and mutual information minimization
                self.enc.zero_grad()
                self.dec.zero_grad()
                data, label = next(dataloader)
                data= data.to(self.device)
                x_hat = self.forward(data) # generate z encodings. defines the 'true' distribution
                l_rec = self.recon_loss(x_hat, data)

                var_zs_agg += self.z_phi.var(dim=0).detach()
                
                # compute loss on the latent space
                l_dtc, l_kl, l_eep = self.latent_loss(self.z_phi)

                loss = l_rec + (info_wgt*l_eep) + (kl_wgt*l_kl) # want to maximize log probability of reconstruction and BCE criterion of information
                loss.backward() # maximize this measure
                self.AE_optim.step()
                l_AE_info_agg += l_dtc.item()
                l_AE_rec_agg += l_rec.item()
                l_Z_kl_agg += l_kl.item()
                mse_recons_agg += 0.
            
            loss_Ds[g] = loss_Ds_agg / (batch_per_group * k)
            dz_agg = dz_agg / (batch_per_group * k)
            dgz_agg = dgz_agg / (batch_per_group * k)
            loss_AE_info[g] = l_AE_info_agg / batch_per_group
            loss_AE_rec[g] = l_AE_rec_agg / batch_per_group
            loss_Z_kl[g] = l_Z_kl_agg / batch_per_group
            mse_recons[g] = mse_recons_agg / batch_per_group
            var_zs[g] = var_zs_agg / batch_per_group

            print("\tD: {:1.3f}\tDT: {:1.3f}\tDF: {:1.3f}\tAE dtc: {:1.3f}\tAE rec loss: {:1.3f}\tAE kl: {:1.3f}".format(\
                    loss_Ds[g].item(), dz_agg, dgz_agg, loss_AE_info[g].item(), loss_AE_rec[g].item(), loss_Z_kl[g].item()))


            # create latent plot and save
            if isinstance(dataset, BeamSynthData) and rec_ls:
                create_latent_image(self, dataset, g+1)

        end_time = time.time()

        print('\nTraining Time: {:.2f}'.format(end_time - start_time))

        return loss_Ds.detach(), loss_AE_info.detach(), loss_AE_rec.detach(), loss_Z_kl.detach(), mse_recons.detach(), var_zs.detach()

    def latent_loss(self, z_phi, unif_samples=50):
        # iterate through each latent variable and calcualte its contribution to sum_info loss & EEP loss
        eep = 0.
        sinfo = 0.
        for i in range(self.z_dim):
            # run tile_fake on z to get (unif_samples x batch_size, z_dim)
            zi_zj = tile_fake(z_phi, i, unif_samples, unif_bnd=bnd)
            # run through the i-th discriminator
            d_zi_zj = torch.sigmoid(self.discs[i](zi_zj))
            
            info_func, h_zi, h_zi_zj = information_functional(d_zi_zj, self.uniform_mult/(2.*bnd), \
                clamp=True, unif_samples=unif_samples, info_only=False)

            sinfo += info_func
            eep += info_func*((2.*h_zi).exp_().detach_())
            
        eep /= (3.14159265359 * 2.7182818284)

        l_kl = torch.zeros(1).to(z_phi)

        return sinfo, l_kl, eep

   


