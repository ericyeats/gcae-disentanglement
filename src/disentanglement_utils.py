import torch
from beamsynth_data import BeamSynthData
import math
import time

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

cmap = get_cmap('tab20b')

bnd = 4.
__eps_denom = 0 # seems that this should be small to not interfere with other eps
__eps_clp = 1e-2
__info_clp = 5.

def cmap_colors(max_ind):
    return [cmap(float(h)/float(max_ind)) for h in range(max_ind)]

def prod(shape):
    a = 1
    for s in shape:
        a *= s
    return a


class UnNormalize(object):
    def __init__(self, norm):
        super(UnNormalize, self).__init__()
        self.mean = norm.mean
        self.std = norm.std

    def __call__(self, tensor):
        return self.scale_inorm(tensor).add(self.mean.to(tensor.device))

    def scale_inorm(self, tensor):
        return tensor.mul(self.std.to(tensor.device))

class Normalize(object):
    def __init__(self, mean, std, ndim=2):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        for d in range(ndim):
            self.mean = self.mean.unsqueeze(-1)
            self.std = self.std.unsqueeze(-1)

    def __call__(self, tensor):
        return tensor.sub(self.mean.to(tensor.device)).div(self.std.to(tensor.device))

def gaussian_reconstruction_dist(mu):
    return torch.distributions.Normal(mu, torch.ones_like(mu))

def bernoulli_reconstruction_dist(p):
    return torch.distributions.Bernoulli(probs=p)

class DistributionLoss(torch.nn.Module):

    def __init__(self, recon_distribution):
        super().__init__()
        self.recon_distribution = recon_distribution

    def forward(self, out, exp):
        dist = self.recon_distribution(out)
        log_probs = dist.log_prob(exp)
        loss = (-1.)*log_probs.sum()/out.shape[0]
        return loss
    
beam_s2s2_norm = Normalize((0.318,), (0.4168), ndim=1)
beam_s2s2_inorm = UnNormalize(beam_s2s2_norm)
    
celeba_norm = Normalize((0.5337, 0.4157, 0.3562), (0.2956, 0.2581, 0.2477))
celeba_inorm = UnNormalize(celeba_norm)

dsprites_norm = Normalize((0.0429,), (0.2026,))
dsprites_inorm = UnNormalize(dsprites_norm)

mnist_norm = Normalize((0.1301,), (0.3074,))
mnist_inorm = UnNormalize(mnist_norm)

def multi_t(a, _f, _t):
    assert _f >= 0
    assert _t >= 0
    assert _f < a.dim()
    assert _t < a.dim()
    while(_f != _t):
        if _f < _t:
            a = a.transpose(_f, _f+1)
            _f += 1
        else:
            a = a.transpose(_f-1, _f)
            _f -= 1
    return a

def grad_mask(_z, i):
    # expect _z to be (batch_size, z_dim)
    # i is the index (< z_dim) to replace with detached copy
    _z_clone = _z.clone()
    _z_clone[..., i] = _z[..., i].detach()
    return _z_clone

def grad_mask_inv(_z, i):
    assert _z.dim() == 2
    assert i < _z.shape[1] and i >= 0
    _z_clone = _z.clone()
    _z_clone[..., 0:i] = _z[..., 0:i].detach()
    _z_clone[..., i+1:_z.shape[1]] = _z_clone[..., i+1:_z.shape[1]].detach()
    return _z_clone

def mask_and_noise(_z, i):
    # expect _z to be (batch_size, z_dim)
    # i is the index (< z_dim) to replace with Gaussian noise
    _z_clone = _z.clone()
    _z_clone[..., i] = torch.randn((_z.shape[0],), device=_z.device)
    return _z_clone

def mask(_z, i):
    # expect _z to be (batch_size, z_dim)
    # i is the index (< z_dim) to replace with Gaussian noise
    _z_clone = _z.clone()
    _z_clone[..., i] = torch.empty((_z.shape[0],), device=_z.device).uniform_(-bnd, bnd)
    return _z_clone

def insert_fake(_z, _z_f, i):
    # _z is the original z batch of size (batch_size, z_dim)
    # _z_f is the batch of fake vectors (batch_size)
    # i is the index (< z_dim) to insert the fake data
    _z_clone = _z.clone()
    _z_clone[..., i] = _z_f
    return _z_clone

def one_cold(i, n, dev='cpu'):
    ones = torch.ones(n)
    ones[i] = 0
    return ones.to(dev)

def agg_list(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
        a[i] += b[i]

def kl_estimate(d_z, rec=False):
    return log_ratio(d_z, rec).mean()

def log_ratio(d_z, rec=False):
    kl_est = ratio(d_z, rec)
    kl_est = torch.log(kl_est)
    return kl_est

def ratio(d_z, rec=False):
    return d_z / (1. - d_z + __eps_denom) if not rec else (1. - d_z) / (d_z + __eps_denom)
    #return d_z / (1. - d_z) if not rec else (1. - d_z) / d_z

# def E_log_density_est_unif(d_z, den, clamp=True):
#     rat = den*ratio(d_z)
#     ratlog = (rat.log()).where(rat >= __eps_clp, torch.zeros_like(rat)) if clamp else rat.log()
#     return ratlog.mean()

# # assume we have uniformly distributed samples for d_z
# def entropy_functional_deriv(d_z, den, clamp=True):
#     rat = den*ratio(d_z)
#     ratlog = (rat.log()).where(rat >= __eps_clp, torch.zeros_like(rat)) if clamp else rat.log()
#     return -(2.*bnd)*((ratlog + 1.).detach_()*rat).mean()

def tile_fake(z, ind, unif_samples=20, unif_bnd=3.):
    bsize = z.shape[0]
    z = z.tile(unif_samples, 1)
    u = torch.empty((unif_samples, 1), device=z.device).uniform_(-unif_bnd, unif_bnd).tile(1, bsize).view(-1)
    z = insert_fake(z, u, ind)
    return z

# # assume d_z has been "tile_faked" and pushed through the discriminator such that it is unif_samples x batch_size
# def entropy_functionals(d_z, den, clamp=True, unif_samples=20):
#     rat = den*ratio(d_z).view(unif_samples, -1)
#     ratlog = (rat.log()).where(rat >= __eps_clp, torch.zeros_like(rat)) if clamp else rat.log()
#     E_zj_rat = rat.mean(dim=1)
#     E_zj_ratlog = (E_zj_rat.log()).where(E_zj_rat >= __eps_clp, torch.zeros_like(E_zj_rat)) if clamp else E_zj_rat.log()
#     marg = (-2.*bnd)*(E_zj_rat*E_zj_ratlog).mean()
#     cond = (-2.*bnd)*(rat*ratlog).mean()
#     return marg, cond

def information_functional(d_z, den, clamp=True, unif_samples=20, info_only=True):
    rat = den*ratio(d_z).view(unif_samples, -1) # (uniform_samples, batch_size)
    E_zj_rat = rat.mean(dim=1) # to get the marginal for each uniform sample (uniform_samples,)
    ratlog = (rat.log()).where(E_zj_rat.view(-1, 1).tile(1, rat.shape[1]) >= __eps_clp, torch.zeros_like(rat)) if clamp else rat.log()
    E_zj_ratlog = (E_zj_rat.log()).where(E_zj_rat >= __eps_clp, torch.zeros_like(E_zj_rat)) if clamp else E_zj_rat.log()
    info_func = ((2.*bnd)*rat*(ratlog - E_zj_ratlog.view(-1, 1).tile(1, ratlog.shape[1])).detach_().clamp_(-__info_clp, __info_clp)).mean()
    if info_only:
        return info_func
    else:
        h_zi_func = -1.*((2.*bnd)*E_zj_rat*E_zj_ratlog).mean().detach_()
        h_zi_zj_func = -1.*((2.*bnd)*rat*ratlog).mean().detach_()
        return info_func, h_zi_func, h_zi_zj_func


def log_prob_est(d_z, den):
    crit = torch.nn.BCEWithLogitsLoss()
    return crit(d_z, torch.zeros_like(d_z)) - crit(d_z, torch.ones_like(d_z)) + torch.log(torch.tensor(den)).to(d_z.device)

def record_latent_space(dgae, dataset, n_dgf=2, n_samples=None):
    if n_samples is None: n_samples = len(dataset)
    latent_storage = torch.zeros((n_samples, dgae.z_dim))
    dgf_storage = torch.zeros((n_samples, n_dgf))
    indices = torch.randperm(len(dataset))[:n_samples]
    # iterate through the training dataset to create scatter plots of the latent space
    with torch.no_grad():
        for i, d_ind in enumerate(indices):
            data, label = dataset[d_ind]
            data  = data.to(dgae.device).unsqueeze_(0) # insert the batch dimension
            out = dgae(data)
            z = dgae.z.cpu()
            latent_storage[i] = z
            dgf_storage[i] = label[:n_dgf].clone()
    
    return latent_storage, dgf_storage

def mig_metric(dgae, train_dataset, n_dgf, dgf_values, n_bins, tols=None):
    latent_storage, dgf_storage = record_latent_space(dgae, train_dataset, n_dgf)
    z_dim = dgae.z_dim
    ls_min, ls_max = latent_storage.min(), latent_storage.max()
    # latent_storage: n_examples x z_dim
    # dgf_storage: n_examples x n_dgf
    # compute the marginal distributions of the latents and compute their maginal entropies
    _marg_hists = []
    _marg_entropies = torch.zeros(z_dim)
    for _i_z in range(z_dim):
        hist_counts, hist_bins = torch.histogram(latent_storage[..., _i_z], bins=torch.linspace(ls_min - 0.05, ls_max + 0.05, n_bins+1))
        _marg_hists.append((hist_counts, hist_bins))
        _marg_probs = hist_counts/hist_counts.sum()
        _marg_entropies[_i_z] = -1.*(_marg_probs*_marg_probs.log()).nan_to_num().sum()
    # iterate through the factors of variation
    agg_mig = 0.0
    n_nonzero_dgf = 0
    for _i_dgf in range(n_dgf):
        # iterate through the possible values for this data generating factor.
        # need information storage for each latent dim
        temp_info = torch.zeros(z_dim)
        _n_v_dgf = len(dgf_values[_i_dgf])
        if _n_v_dgf > 1:
            n_nonzero_dgf += 1
            _h_vk = 0.
            for _v_dgf in dgf_values[_i_dgf]:
                # recover the latent encodings corresponding to this value of the data generating factor
                if tols is None:
                    _v_dgf_mask = dgf_storage[..., _i_dgf].isclose(torch.full_like(dgf_storage[..., _i_dgf], _v_dgf)) # should be n_examples
                else:
                    assert len(tols) == n_dgf
                    _v_dgf_mask = dgf_storage[..., _i_dgf].isclose(torch.full_like(dgf_storage[..., _i_dgf], _v_dgf), atol=tols[_i_dgf], rtol=0.) # should be n_examples
                _p_vk = float(_v_dgf_mask.sum().item()) / float(_v_dgf_mask.numel())
                if _p_vk == 0:
                    continue
                _h_vk -= _p_vk*torch.log(torch.tensor(_p_vk))
                # use this to select latent encodings corresponding to this dgf_value
                _lat_encs_v = latent_storage[_v_dgf_mask] # should be n_examples_dgf_val, z_dim
                # iterate through the z dimension to begin calculating mutual information
                for _i_z in range(z_dim):
                    # create a histogram of a fixed number of bins to quantize p(z_i|v_k=_v_dgf) for Info est
                    cond_hist_vals, _ = torch.histogram(_lat_encs_v[..., _i_z], _marg_hists[_i_z][1])
                    #print(_lat_encs_v)
                    #print(cond_hist_vals)
                    # compute information contribution 
                    _p_z_gvks = cond_hist_vals/cond_hist_vals.sum()
                    _z_marg_hist = _marg_hists[_i_z][0]
                    _p_z_margs = _z_marg_hist/_z_marg_hist.sum()
                    temp_info[_i_z] += _p_vk*(_p_z_gvks*torch.log(_p_z_gvks/_p_z_margs)).nan_to_num().sum()
            #norm_info = torch.zeros(z_dim).where(_marg_entropies <= 0., temp_info/_marg_entropies)
            norm_info = temp_info/_h_vk
            # find the max and next-max infos
            _max_info, ind = torch.max(norm_info, dim=0)
            print(norm_info, _max_info, ind)
            norm_info[ind] = 0.
            _next_max_info = norm_info.max()
            agg_mig += _max_info - _next_max_info
    if n_nonzero_dgf != 0:
        agg_mig /= n_nonzero_dgf
    return agg_mig

def factor_metric_sample_ind(batch_size, dgf_lens):
    # randomly choose a factor to hold constant
    c_fac_ind = torch.randint(dgf_lens.shape[0])
    c_fac_val_ind = torch.randint(dgf_lens[c_fac_ind])
    # generate random samples for each factor, then cat
    factor_samples = []
    for fac_ind in range(dgf_lens.shape[0]):
        factor_samples.append(torch.randint(dgf_lens[fac_ind], size=(batch_size, 1)))
    samples = torch.cat(factor_samples, dim=1)
    samples[..., c_fac_ind] = c_fac_val_ind # make these constant
    return samples

def get_lat_storage(dgae, dataset, n_dgf, n_data=5000, batch_size=64):
    assert (n_data <= len(dataset))
    # create shuffled splits of the dataset
    latent_storage = torch.empty((n_data, dgae.z_dim)).to(dgae.device)
    factor_storage = torch.empty((n_data, n_dgf)).to(dgae.device)
    dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True))
    base_index = 0
    with torch.no_grad():
        while base_index < n_data:
            data, lab = next(dataloader_iter)
            out = dgae(data.to(dgae.device))
            z = dgae.z
            next_index = min(base_index+batch_size, n_data)
            latent_storage[base_index:next_index] = z[:next_index-base_index]
            factor_storage[base_index:next_index] = lab[:next_index-base_index, :n_dgf]
            base_index = next_index
    return latent_storage, factor_storage

def factor_metric(dgae, dataset, n_dgf, n_calibrate=10000, n_train=5000, n_test=1000, batch_size=64):

    latent_storage, _ = get_lat_storage(dgae, dataset, n_dgf, n_calibrate, batch_size)
    lat_std = latent_storage.std(dim=0, unbiased=True)
    var_mask = (lat_std.square() >= 0.05).to(dgae.device)

    tally = torch.zeros((dgae.z_dim, n_dgf))
    # generate training points for the majority-vote classifier
    for i in range(n_train):
        # push n_train batches (with one factor fixed each) through the encoder
        # excluding collapsed dimensions (var >= 0.05), which one has the smallest normalized variance?
        const_lat, data = dataset.sample_latent_factor_metric(batch_size)
        out = dgae(data.to(dgae.device))
        z_scl = (dgae.z / lat_std).where(var_mask.unsqueeze(0).tile(batch_size, 1), dgae.z)
        z_scl_var = (z_scl.var(dim=0)).where(var_mask, torch.full((dgae.z_dim,), torch.inf, device=dgae.device))
        lat_ind = z_scl_var.min(dim=0)[1].item()
        tally[lat_ind, const_lat] += 1.
    
    # construct MVC := for each latent var, which is the most-voted k?
    mvc = tally.max(dim=1)[1] # (z_dim,) - stores the index of MVC dgf

    predictions = torch.zeros(n_test, dtype=torch.long)
    ground_truth = torch.zeros(n_test, dtype=torch.long)
    # generate testing points for the MVC
    for i in range(n_test):
        # push n_train batches (with one factor fixed each) through the encoder
        # excluding collapsed dimensions (var >= 0.05), which one has the smallest normalized variance?
        const_lat, data = dataset.sample_latent_factor_metric(batch_size)
        out = dgae(data.to(dgae.device))
        z_scl = (dgae.z / lat_std).where(var_mask.unsqueeze(0).tile(batch_size, 1), dgae.z)
        z_scl_var = (z_scl.var(dim=0)).where(var_mask, torch.full((dgae.z_dim,), torch.inf, device=dgae.device))
        lat_ind = z_scl_var.min(dim=0)[1].item()
        predictions[i] = mvc[lat_ind]
        ground_truth[i] = const_lat
    
    return (predictions == ground_truth).sum() / n_test
            
def dci_disentanglement_metric(dgae, dataset, n_dgf, n_test=10000, batch_size=64):
    # train n_dgf gradient boosted trees
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    train_data, train_lab = get_lat_storage(dgae, dataset, n_dgf, n_test, batch_size)
    importance_mat = np.zeros((dgae.z_dim, n_dgf), dtype=np.float64)
    for i in range(n_dgf):
        lab_i = train_lab[..., i].cpu().numpy()
        le = LabelEncoder()
        le.fit(lab_i)
        if len(le.classes_) > 1:
            lab_i = le.transform(lab_i)
            model = GradientBoostingClassifier()
            model.fit(train_data.cpu().numpy(), lab_i)
            importance_mat[:, i] = np.abs(model.feature_importances_)
        else:
            importance_mat[:, i] = np.full((dgae.z_dim,), 1./dgae.z_dim)
    # compute k-entropy along the columns
    importance_mat = torch.tensor(importance_mat).t()
    norm_import_mat = importance_mat / importance_mat.sum(dim=0) + 1e-11 # following locatello
    assert norm_import_mat.sum(dim=0).allclose(torch.ones(dgae.z_dim, dtype=torch.double))
    entropies = -1.*(norm_import_mat*(norm_import_mat.log()/np.log(n_dgf))).sum(dim=0)
    rel_import = importance_mat.sum(dim=0) / importance_mat.sum()
    return (rel_import*(1 - entropies)).sum()

def sap_metric(dgae, dataset, n_dgf, is_continuous, n_train=5000, n_test=1000, batch_size=64):
    from sklearn import svm
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    latent_storage, factor_storage = get_lat_storage(dgae, dataset, n_dgf, n_train+n_test, batch_size)
    latent_train, factor_train = latent_storage[:n_train], factor_storage[:n_train]
    latent_test, factor_test = latent_storage[n_train:], factor_storage[n_train:]
    # collect score_matrix entries
    score_matrix = torch.zeros((dgae.z_dim, n_dgf))
    for i in range(dgae.z_dim):
        for j in range(n_dgf):
            lats = latent_test[..., i]
            facs = factor_test[..., j]
            if is_continuous[j]:
                cov_zi_vj = torch.cov(torch.cat((lats.view(1, -1), facs.view(1, -1)), dim=0))
                cov_z_v = cov_zi_vj[0, 1]**2
                var_z = cov_zi_vj[0, 0]
                var_v = cov_zi_vj[1, 1]
                if var_z > 1e-12:
                    score_matrix[i, j] = cov_z_v / (var_z * var_v)
                else:
                    score_matrix[i, j] = 0.

            else:
                lats_train = latent_train[..., i].view(-1, 1).cpu().numpy()
                facs_train = factor_train[..., j].cpu().numpy()
                # use labelencoder
                le = LabelEncoder()
                le.fit(facs_train)
                if len(le.classes_) > 1:
                    # discrete variables need to be fit to a classifier
                    classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                    classifier.fit(lats_train, le.transform(facs_train))
                    pred = classifier.predict(lats.view(-1, 1).cpu().numpy())
                    score_matrix[i, j] = np.mean(pred == le.fit(facs))
                else: 
                    score_matrix[i, j] = 1.
    # score matrix is complete
    # for each latent factor, take the top two scores. the average difference is the SAP score
    max1, max1_ind = score_matrix.max(dim=1)
    score_matrix2 = score_matrix.clone()
    score_matrix2[list(range(dgae.z_dim)), max1_ind] = 0.
    max2 = score_matrix2.max(dim=1)[0]
    return (max1 - max2).mean()


def collect_lps(device, model, dataset, ds_lp_func, batch_size):
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        x_lps = torch.zeros(len(dataset), device=device)
        x_lp_ests = torch.zeros(len(dataset), device=device)
        for i, (data, lab) in enumerate(dataloader):
            data, lp = data.to(device), ds_lp_func(lab.to(device))
            x_hat, lp_est = model.log_density_est(data)
            x_lps[i*batch_size:i*batch_size+lp.shape[0]] = lp
            x_lp_ests[i*batch_size:i*batch_size+lp.shape[0]] = lp_est
    return x_lps, x_lp_ests

def kldiv_metric(device, model, dataset, ds_lp_func, batch_size=50):
    x_lps, x_lp_ests = collect_lps(device, model, dataset, ds_lp_func, batch_size=50)
    return x_lps.mean(), x_lp_ests.mean()

def sim_metric(device, model, dataset, ds_lp_func, batch_size=50):
    x_lps, x_lp_ests = collect_lps(device, model, dataset, ds_lp_func, batch_size)
    # shift to isolate relative densities
    x_lps = x_lps - x_lps.mean()
    x_lp_ests = x_lp_ests - x_lp_ests.mean()
    return (x_lps*x_lp_ests).sum()/(x_lps.norm(p=2)*x_lp_ests.norm(p=2))
            
def create_latent_image(dgae, train_dataset, grp):
    view_alt=5
    view_ang=90
    alpha=0.5

    zlim = None

    latent_storage, dgf_storage = record_latent_space(dgae, train_dataset, 2)

    fig = plt.figure(figsize=(10, 4))
    ax = None
    ax = fig.add_subplot(121, projection='3d')

    colors = cmap_colors(dgae.z_dim)

    for i in range(dgae.z_dim):
        ax.scatter(dgf_storage[..., 1], dgf_storage[..., 0], latent_storage[..., i], label='L{}'.format(i+1), alpha=alpha, color=colors[i])

    ax.view_init(view_alt, view_ang)
    ax.set_xlabel('S2_duty_cycle')
    ax.set_ylabel('S2_frequency')
    ax.set_zlabel('Latent Activation')
    ax.set_zlim(zlim)
    #ax.legend()

    ax = fig.add_subplot(122, projection='3d')

    for i in range(dgae.z_dim):
        ax.scatter(dgf_storage[..., 1], dgf_storage[..., 0], latent_storage[..., i], label='L{}'.format(i+1), alpha=alpha, color=colors[i])

    ax.view_init(view_alt, view_ang + 45)
    ax.set_xlabel('S2_duty_cycle')
    ax.set_ylabel('S2_frequency')
    ax.set_zlabel('Latent Activation')
    if zlim is not None:
        ax.set_zlim(zlim)
    ax.legend()

    plt.savefig('./figs/beamsynthesis/z_g{}.png'.format(grp))
    plt.close()


class AutoEncoder(torch.nn.Module):

    def __init__(self, enc, dec, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.enc = enc
        self.dec = dec
        self.z = None
        self.device = 'cpu'

    def forward(self, x):
        self.z = self.enc(x)
        return self.dec(self.z)

    def train(self):
        self.enc.train()
        self.dec.train()

    def eval(self):
        self.enc.eval()
        self.dec.eval()

class Adversarial_AutoEncoder(AutoEncoder):

    """
    enc, dec, z_dim, disc_arch
    """

    def __init__(self, enc, dec, z_dim, disc_arch):
        super().__init__(enc, dec, z_dim)
        self.disc = disc_arch(z_dim)

    def init_optim_objects(self, lr_D=0.001, lr_AE=0.001, sample_per_k=2):
        self.D_optim = torch.optim.Adam(self.disc.parameters(), lr=lr_D, betas=(0.5, 0.9))
        self.AE_optim = torch.optim.Adam(list(self.enc.parameters()) + list(self.dec.parameters()), lr=lr_AE)
        self.criterion = torch.nn.BCELoss()
        self.recon_loss = torch.nn.MSELoss()
        self.sample_per_k = sample_per_k

    def train(self):
        super().train()
        self.disc.train()

    def eval(self):
        super().eval()
        self.disc.eval()

    def to(self, device):
        self.disc = self.disc.to(device)
        self.enc = self.enc.to(device)
        self.dec = self.dec.to(device)
        self.device = device
        return self

    def real_sample(self, z):
        return z

    def fake_sample(self, z):
        return torch.randn_like(z) # prior distribution

    def train_discriminator(self, dataloader, loss_D_agg, dz_agg, dgz_agg, loss_preds_agg):
        # train the discriminator on real data
        self.disc.zero_grad()
        # define the joint data distribution
        data, label = next(dataloader)
        data = data.to(self.device)
        out = self.forward(data) # generate z encodings. defines the 'true' distribution
        z = self.z.detach()

        z_real = self.real_sample(z)

        pred_real = torch.sigmoid(self.disc(z_real))
        dz_agg += pred_real.mean().item()
        expec_real = torch.ones_like(pred_real)
        l_diss = self.criterion(pred_real, expec_real.detach())

        # define the fake data distribution
        data, label = next(dataloader)
        data = data.to(self.device)
        out = self.forward(data)
        z = self.z.detach()

        z_fake = self.fake_sample(z)

        pred_fake = torch.sigmoid(self.disc(z_fake))
        dgz_agg += pred_fake.mean().item()
        expec_fake = torch.zeros_like(pred_fake)
        l_diss += self.criterion(pred_fake, expec_fake.detach())
        

        l_diss.backward()
        self.D_optim.step()

        loss_D_agg += l_diss.item()

        return loss_D_agg, dz_agg, dgz_agg, loss_preds_agg

    def latent_loss(self, z, jsd=False):
        # calculate divergence contribution from data distribution to merge
        z_real = self.real_sample(z) # gives back z
        l_info = None
        pred_real = torch.sigmoid(self.disc(z_real))

        if jsd:
            expec_real = torch.zeros_like(pred_real)
            l_info = self.criterion(pred_real, expec_real.detach())
        else:
            l_info = kl_estimate(pred_real)

        return l_info, 0.0


    def fit(self, dataset, n_group, batch_per_group=1000, batch_size=64, k=1, info_wgt=0.0, kl_wgt=1.0, ws_disc=0, rec_ls=False):
        start_time = time.time()
        assert info_wgt >= 0.
        self.train()

        # set up loss storages
        loss_Ds = torch.zeros(n_group)
        loss_AE_info = torch.zeros(n_group)
        loss_AE_rec = torch.zeros(n_group)
        loss_AE_kl = torch.zeros(n_group)
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
            _, _, _, _ = self.train_discriminator(dataloader, 0., 0., 0., 0.)

        for g in range(n_group):
            loss_D_agg = 0.
            dz_agg = 0.
            dgz_agg = 0.
            l_AE_info_agg = 0.
            l_AE_rec_agg = 0.
            l_AE_kl_agg = 0.
            var_zs_agg = torch.zeros(self.z_dim).to(self.device)

            for b in range(batch_per_group):
                print("\rGroup: {}\t{:2.0f}%".format(g+1, 100*(b+1)/batch_per_group), end="")

                # begin by training the Discriminators/Generators for K steps
                for _ in range(k):

                    loss_D_agg, dz_agg, dgz_agg, l_AE_kl_agg = self.train_discriminator(dataloader, loss_D_agg, dz_agg, dgz_agg, l_AE_kl_agg)

                # train the AE for reconstruction and mutual information minimization
                self.enc.zero_grad()
                self.dec.zero_grad()
                data, label = next(dataloader)
                data= data.to(self.device)
                out = self.forward(data) # generate z encodings. defines the 'true' distribution
                z = self.z

                var_zs_agg += self.z.var(dim=0).detach()
                
                # define loss on the 
                l_latent, prior_loss = self.latent_loss(z)

                l_rec = self.recon_loss(out, data)
                
                loss = (info_wgt)*(l_latent) + (l_rec + kl_wgt*prior_loss) # want to maximize log probability of reconstruction and BCE criterion of information
                loss.backward() # maximize this measure
                self.AE_optim.step()
                l_AE_info_agg += l_latent.item()
                l_AE_rec_agg += l_rec.item()
            
            loss_Ds[g] = loss_D_agg / (batch_per_group * k)
            dz_agg = dz_agg / (batch_per_group * k)
            dgz_agg = dgz_agg / (batch_per_group * k)
            loss_AE_info[g] = l_AE_info_agg / batch_per_group
            loss_AE_rec[g] = l_AE_rec_agg / batch_per_group
            loss_AE_kl[g] = l_AE_kl_agg / batch_per_group
            var_zs[g] = var_zs_agg / batch_per_group

            print("\tD: {:1.3f}\tDT: {:1.3f}\tDF: {:1.3f}\tAE info loss: {:1.3f}\tAE rec loss: {:1.3f}\tAE kl loss: {:1.3f}".format(\
                    loss_Ds[g].item(), dz_agg, dgz_agg, loss_AE_info[g].item(), loss_AE_rec[g].item(), loss_AE_kl[g].item()))


            # create latent plot and save
            if isinstance(dataset, BeamSynthData) and rec_ls:
                create_latent_image(self, dataset, g+1)

        end_time = time.time()

        print('\nTraining Time: {:.2f}'.format(end_time - start_time))
        return loss_Ds.detach(), loss_AE_info.detach(), loss_AE_rec.detach(), loss_AE_kl.detach(), mse_recons.detach(), var_zs.detach()


class FACTOR_VAE(Adversarial_AutoEncoder):

    def __init__(self, enc, dec, z_dim, disc_arch, out_act=torch.nn.Identity()):
        super().__init__(enc, dec, z_dim, disc_arch)
        self.out_act = out_act

    def init_optim_objects(self, lr_D=0.001, lr_AE=0.001, sample_per_k=2, recon_loss=gaussian_reconstruction_dist):
        super().init_optim_objects(lr_D, lr_AE, sample_per_k)
        self.recon_loss = DistributionLoss(recon_loss)

    def forward(self, x, n_samp=1):
        self.z, self.z_log_var = self.enc(x)
        self.enc_dist = torch.distributions.Normal(self.z, (0.5*self.z_log_var).exp())
        self.z_sample = self.enc_dist.rsample((n_samp,))
        if n_samp == 1:
            self.z_sample = self.z_sample.view(-1, self.z_dim)
        x_hat = self.out_act(self.dec(self.z_sample.view(x.shape[0]*n_samp, self.z_dim)).view((n_samp,) + x.shape))
        return x_hat if n_samp > 1 else x_hat.view(x.shape)

    def train_discriminator(self, dataloader, loss_D_agg, dz_agg, dgz_agg, l_AE_kl_agg):
        # train the discriminator on real data
        self.disc.zero_grad()
        # define the joint data distribution
        data, label = next(dataloader)
        data = data.to(self.device)
        out = self.forward(data) # generate z encodings. defines the 'true' distribution
        z = self.z_sample.detach()

        z_real = self.real_sample(z)

        pred_real = torch.sigmoid(self.disc(z_real))
        dz_agg += pred_real.mean().item()
        expec_real = torch.ones_like(pred_real)
        l_diss = self.criterion(pred_real, expec_real.detach()).mean()

        # define the fake data distribution
        data, label = next(dataloader)
        data = data.to(self.device)
        out = self.forward(data)
        z = self.z_sample.detach()

        z_fake = self.fake_sample(z)

        pred_fake = torch.sigmoid(self.disc(z_fake))
        dgz_agg += pred_fake.mean().item()
        expec_fake = torch.zeros_like(pred_fake)
        l_diss += self.criterion(pred_fake, expec_fake.detach()).mean()

        l_diss /= 2.
        

        l_diss.backward()
        self.D_optim.step()

        loss_D_agg += l_diss.item()

        return loss_D_agg, dz_agg, dgz_agg, l_AE_kl_agg

    def fake_sample(self, z): # shuffle everything to get the marginals
        shuff_inds = [torch.randperm(z.shape[0]).to(self.device) for _ in range(self.z_dim)]

        z_fake = torch.empty_like(z)
        for _z_ind in range(self.z_dim):
            z_fake[..., _z_ind] = z[..., _z_ind][shuff_inds[_z_ind]]
        
        return z_fake

    def kl_div_prior(self, z_samp, reduce=True):
        # z_samp is (n_samp, batch_size, z_dim)
        prior_dist = torch.distributions.Normal(torch.zeros_like(z_samp), torch.ones_like(z_samp))
        prior_log_prob = prior_dist.log_prob(z_samp).sum(dim=-1)
        post_log_prob = self.enc_dist.log_prob(z_samp).sum(dim=-1) # independent
        kl_div = (post_log_prob - prior_log_prob).logsumexp(dim=0) - torch.tensor(z_samp.shape[0]).log()
        if reduce:
            kl_div = kl_div.mean()
        return kl_div

    def elbo(self, x, n_samp=1, reduce=True):
        # x: (batch_size,) + x_shape
        # n_samp: number of samples for elbo
        x_hat = self.forward(x, n_samp) # get encoding distribution parameters
        # print(x_hat.shape)
        prior_dist = torch.distributions.Normal(torch.zeros_like(self.z_sample), torch.ones_like(self.z_sample)) # p(z)
        recon_dist = self.recon_dist(x_hat) # p(x|z)
        # for each sample in n_sample, compute p(z), q(z|x), p(x|z)
        log_pz = prior_dist.log_prob(self.z_sample).sum(dim=-1) # (n_sample, batch_size)
        log_qz_x = self.enc_dist.log_prob(self.z_sample).sum(dim=-1) # (n_sample, batch_size)
        log_px_z = recon_dist.log_prob(x.unsqueeze(0).expand_as(x_hat)).sum(dim=tuple(range(x_hat.dim())[2:])) # (n_sample, batch_size)
        elbo = log_pz + log_px_z - log_qz_x
        # need to take average over n_sample
        elbo = torch.logsumexp(elbo, dim=0) - torch.tensor(n_samp, device=elbo.device).log() # (batch_size,)
        if reduce:
            elbo = elbo.mean()
        return elbo

    def marginal_log_likelihood(self, x, n_samp=1):
        return self.elbo(x, n_samp, False)

    def log_density_est(self, x):
        return x, self.marginal_log_likelihood(x, n_samp=100)

    def latent_loss(self, z):
        d_z = torch.sigmoid(self.disc(self.z_sample))
        tc_loss = torch.log((d_z + 1e-5)/(1. - d_z + 1e-5)).mean() # add small eps to avoid log 0 and division by zero

        # z is mu of latent representation
        # calculate kl divergence with gaussian prior distribution
        prior_div_loss = self.kl_div_prior(self.z_sample, reduce=True)

        return tc_loss, prior_div_loss

class B_TCVAE(FACTOR_VAE):

    def fit(self, dataset, n_group, batch_per_group=1000, batch_size=64, k=1, info_wgt=0.0, kl_wgt=1.0, ws_disc=0, rec_ls=False):
        self.dataset_len = len(dataset)
        return super().fit(dataset, n_group, batch_per_group, batch_size, k, info_wgt, kl_wgt, ws_disc, rec_ls)

     # FROM RTQ CHEN at https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py -----------
    def qz_estimate(self, _z_s, _z_mu, _z_log_var, _dataset_size):
        """
        _z_s: samples of z of shape (batch_size, z_dim)
        _z_mu: mu parameter of encoded distribution of shape (batch_size, z_dim)
        _z_log_var: variance parameter of encoded distribution of shape (batch_size, z_dim)
        _dataset_size: len of dataset
        """
        M = _z_s.shape[0]
        # iterate through the sample dimension
        running_sum = 0.
        for i in range(M):
            _z_s_expand = _z_s[i].unsqueeze(0).expand(_z_s.shape)
            # now compute log prob against mu, log_var
            log_probs = compute_log_prob(_z_s_expand, _z_mu, torch.exp(_z_log_var*0.5))
            running_sum += logsumexp(log_probs)
        running_sum /= M
        running_sum -= torch.log(M*_dataset_size)
        return running_sum
    
    def qz_estimate_rtqc(self, _z_s, _z_mu, _z_log_var):
        batch_size = _z_s.shape[0]
        _z_var = torch.exp(_z_log_var*0.5)
        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = compute_log_prob(_z_s.view(batch_size, 1, self.z_dim), \
                                  _z_mu.view(1, batch_size, self.z_dim), _z_var.view(1, batch_size, self.z_dim))
        # minibatch weighted sampling
        logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * self.dataset_len)).sum(1)
        logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * self.dataset_len))
        return logqz_prodmarginals, logqz

    def latent_loss(self, z):
        logqz_condx = compute_log_prob(self.z_sample, self.z, torch.exp(self.z_log_var*0.5)).sum(1)
        logqz_prodmarginals, logqz = self.qz_estimate_rtqc(self.z_sample, self.z, self.z_log_var)
        logpz = compute_log_prob(self.z_sample).sum(1) # sum across the z_dim dimension
        mi_term = (logqz_condx - logqz)
        tc_term = (logqz - logqz_prodmarginals)
        skl_term = (logqz_prodmarginals - logpz)
        return tc_term.mean(), (mi_term + skl_term).mean()

def compute_log_prob(sample, dstr_mu=None, dstr_sig=None):
    if dstr_mu is None:
        dstr_mu=torch.zeros_like(sample)
    if dstr_sig is None:
        dstr_sig=torch.ones_like(sample)
    dstr = torch.distributions.Normal(dstr_mu, dstr_sig)
    lprob = dstr.log_prob(sample)
    return lprob

# From https://github.com/rtqichen/beta-tcvae/blob/master/vae_quant.py
def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, float):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

class DIP_VAE_2(FACTOR_VAE):

    def latent_loss(self, z):
        # calculate the covariance matrix of z samples
        z_cov = torch.cov(self.z_sample.t())
        diag_mask = torch.eye(self.z_dim).to(self.z_sample.device)
        cov_loss = (diag_mask * (z_cov - 1.).square() + (1. - diag_mask) * z_cov.square()).sum()

        # z is mu of latent representation
        # calculate kl divergence with gaussian prior distribution
        prior_div_loss = self.kl_div_prior(self.z_sample, reduce=True)

        return cov_loss, prior_div_loss