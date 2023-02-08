import torch
from torch.nn.init import kaiming_normal_
from disentanglement_utils import prod

def leaky_relu_act():
    return torch.nn.LeakyReLU(0.2)

def elu_act():
    return torch.nn.ELU()

def selu_act():
    return torch.nn.SELU()

def relu_act():
    return torch.nn.ReLU()

def relu_init(lay):
    kaiming_normal_(lay.weight.data, nonlinearity='relu')
    return lay

def default_init(lay):
    return lay

def selu_init(lay):
    kaiming_normal_(lay.weight.data, nonlinearity='linear')
    return lay


def latent_init(linear_module):
    #linear_module.weight.data.div_(linear_module.weight.data.shape[0])
    linear_module.weight.data.div_(2)
    return linear_module


class AE_BASE(torch.nn.Module):

    def __init__(self, z_dim=24, x_shape=(1, 28, 28), act=selu_act, init=selu_init):
        super().__init__()
        self.z_dim = z_dim
        self.x_shape = x_shape
        self.x_dim = prod(x_shape)
        self.act = act
        self.init = init

class MLP_ENC(AE_BASE):

    def __init__(self, z_dim=24, x_shape=(1, 28, 28), act=selu_act, init=selu_init):
        super().__init__(z_dim, x_shape, act, init)
        self.fc_op = torch.nn.modules.Sequential(
            init(torch.nn.Linear(self.x_dim, 1024)),
            act(),
            init(torch.nn.Linear(1024, 1024)),
            act(),
            init(torch.nn.Linear(1024, 512)),
            act(),
            latent_init(init(torch.nn.Linear(512, z_dim)))
        )

    def forward(self, x):
        x = x.view(-1, self.x_dim)
        return self.fc_op(x)

class MLP_ENC_VAE(AE_BASE):

    def __init__(self, z_dim=24, x_shape=(1, 28, 28), act=selu_act, init=selu_init):
        super().__init__(z_dim, x_shape, act)
        self.fc_op = torch.nn.modules.Sequential(
            init(torch.nn.Linear(self.x_dim, 1024)),
            act(),
            init(torch.nn.Linear(1024, 1024)),
            act(),
            init(torch.nn.Linear(1024, 512)),
            act()
        )
        self.fc_mu = latent_init(init(torch.nn.Linear(512, z_dim)))
        self.fc_log_var = init(torch.nn.Linear(512, z_dim))

    def forward(self, x):
        x = x.view(-1, self.x_dim)
        x_inter = self.fc_op(x)
        return self.fc_mu(x_inter), self.fc_log_var(x_inter)

class MLP_DEC(AE_BASE):

    def __init__(self, z_dim=24, x_shape=(1, 28, 28), act=selu_act, init=selu_init):
        super().__init__(z_dim, x_shape, act)
        self.fc_op = torch.nn.modules.Sequential(
            init(torch.nn.Linear(z_dim, 512)),
            act(),
            init(torch.nn.Linear(512, 1024)),
            act(),
            init(torch.nn.Linear(1024, 1024)),
            act(),
            init(torch.nn.Linear(1024, self.x_dim))
        )

    def forward(self, z):
        x = self.fc_op(z).view((-1,) + self.x_shape)
        return x

class COND_ARCH(torch.nn.Module):

    def __init__(self, z_dim=24, hidden_size=512, act=selu_act, init=selu_init):
        super().__init__()
        self.z_dim = z_dim
        self.fc_op = torch.nn.modules.Sequential(
            init(torch.nn.Linear(z_dim, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, 1)),
        )

    def forward(self, z):
        return self.fc_op(z)

class FACTORVAE_COND_ARCH(torch.nn.Module):

    def __init__(self, z_dim=24, hidden_size=1024, act=selu_act, init=selu_init):
        super().__init__()
        self.z_dim = z_dim
        self.fc_op = torch.nn.modules.Sequential(
            init(torch.nn.Linear(z_dim, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, hidden_size)),
            act(),
            init(torch.nn.Linear(hidden_size, 1)),
        )

    def forward(self, z):
        return self.fc_op(z)

