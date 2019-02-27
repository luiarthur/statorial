import torch

# Set default type to float64 (instead of float32)
torch.set_default_dtype(torch.float64)

# Model parameter
class ModelParam():
    def __init__(self, size, m=None, log_s=None):
        if m is None:
            m = torch.randn(size)

        if log_s is None:
            log_s = torch.randn(size)

        self.vp = torch.stack([m, log_s])
        self.vp.requires_grad = True
        self.size = size

    def dist(self):
        return torch.distributions.Normal(self.vp[0], self.vp[1].exp())

    def rsample(self, n=torch.Size([])):
        # NOTE: The same as:
        # self.vp[0] + torch.randn(n) * self.vp[1].exp()
        # Note that the random number is sampled from a (parameter-free)
        # standard normal.
        return self.dist().rsample(n)

    def log_q(self, real):
        return self.dist().log_prob(real).sum()


