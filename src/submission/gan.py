import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    d_loss = None
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    ### START CODE HERE ###
    log_d_x = torch.log(torch.sigmoid(d(x_real)))
    log_d_z = torch.log(1 - torch.sigmoid(d(g(z))))
    d_loss = -log_d_x.mean() - log_d_z.mean()
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    ### START CODE HERE ###
    log_d_z = torch.log(torch.sigmoid(d(g(z))))
    g_loss = - log_d_z.mean()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    ### START CODE HERE ###
    log_d_x = torch.log(torch.sigmoid(d(x_real, y_real)))
    log_d_z = torch.log(1 - torch.sigmoid(d(g(z, y_real), y_real)))
    d_loss = -log_d_x.mean() - log_d_z.mean()
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None

    ### START CODE HERE ###
    log_d_z = torch.log(1 - torch.sigmoid(d(g(z, y_real), y_real)))
    g_loss = - log_d_z.mean()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    ### START CODE HERE ###
    x_fake = g(z)

    d_x = d(x_real)
    d_z = d(x_fake)

    # calc regularization term
    alpha = torch.rand(batch_size, device=device)
    mixed_x = torch.einsum("i,ijkl->ijkl", alpha, x_fake) + torch.einsum("i,ijkl->ijkl", 1-alpha, x_real)
    d_mixed_x = d(mixed_x)
    grad = torch.autograd.grad(
        outputs=d_mixed_x,
        inputs=mixed_x,
        grad_outputs=torch.ones_like(d_mixed_x), # used to get gard for every instance in the batch
        create_graph=True
    )[0]
    f_norm = torch.norm(grad.view(batch_size, -1), dim=1)
    regular = torch.einsum("i,i->i", f_norm-1, f_norm-1)

    d_loss = d_z.mean() - d_x.mean() + 10 * regular.mean()
    return d_loss
    ### END CODE HERE ###
    raise NotImplementedError


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    ### START CODE HERE ###
    d_z = d(g(z))
    g_loss = - d_z.mean()
    return g_loss
    ### END CODE HERE ###
    raise NotImplementedError