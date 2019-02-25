import torch
from torch.distributions import Normal, Gamma
from torch.distributions.log_normal import LogNormal

import numpy as np
import matplotlib.pyplot as plt

import advi

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

def log_prior_plus_logabsdet_J(real_params, params):
    # log prior for beta, evaluated at sampled values for beta
    lp_b = Normal(0, 1).log_prob(real_params['beta']).sum()

    # log prior sig + log jacobian
    lp_log_sig = LogNormal(0, 1).log_prob(params['sig']) + real_params['sig']
    lp_log_sig = lp_log_sig.sum()

    return lp_b + lp_log_sig

def loglike(y, x, params, full_data_size):
    beta = params['beta']
    sig = params['sig']
    return Normal(x.matmul(beta), sig).log_prob(y).mean(0) * full_data_size

def log_q(model_params, real_params):
    out = 0.0
    for key in model_params:
        out += model_params[key].log_q(real_params[key])
    return out

def elbo(y, x, model_params, full_data_size):
    real_params = {}
    for key in model_params:
        real_params[key] = model_params[key].rsample()

    params = {'beta': real_params['beta'],
              'sig': real_params['sig'].exp()}

    out = loglike(y, x, params, full_data_size)
    out += log_prior_plus_logabsdet_J(real_params, params) 
    out -= log_q(model_params, real_params)

    return out

if __name__ == '__main__':
    # set random number generator seeds for reproducibility
    torch.manual_seed(1)
    np.random.seed(0)

    N = 1000
    x = torch.stack([torch.ones(N), torch.randn(N)], -1)
    k = x.shape[1]
    beta = torch.tensor([2., -3.])
    sig = torch.tensor([0.5])
    y = Normal(x.matmul(beta), sig).rsample()

    # Plot data
    plt.scatter(x[:, 1].numpy(), y.numpy()); plt.show()

    model_params = {'beta': advi.ModelParam(size=k), 'sig': advi.ModelParam(size=1)}
    optimizer = torch.optim.Adam([model_params[key].vp for key in model_params], lr=.1)
    elbo_hist = []

    max_iter = 10000
    minibatch_size = 100
    for t in range(max_iter):
        sample_with_replacement = minibatch_size > N
        idx = np.random.choice(N, minibatch_size, replace=sample_with_replacement)
        loss = -elbo(y[idx], x[idx, :], model_params, full_data_size=N) / N
        elbo_hist.append(-loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (t + 1) % (max_iter / 10) == 0:
            print('iteration: {}/{} | elbo: {}'.format(t + 1, max_iter, elbo_hist[-1]))


    # Inspect posterior
    nsamps = 1000
    beta_post = model_params['beta'].rsample([nsamps]).detach().numpy()
    sig_post = model_params['sig'].rsample([nsamps]).exp().detach().numpy()

    plt.plot(elbo_hist)
    plt.show()

    plt.hist(beta_post[:, 0])
    plt.hist(beta_post[:, 1])
    plt.show()

    plt.hist(sig_post)
    plt.axvline(sig_post.mean(), color='black', linestyle='--')
    plt.axvline(np.percentile(sig_post, 2.5), color='black', linestyle='--')
    plt.axvline(np.percentile(sig_post, 97.5), color='black', linestyle='--')
    plt.show()
