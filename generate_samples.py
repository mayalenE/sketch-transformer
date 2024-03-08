import svgwrite
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, OneHotCategorical, Categorical

# Training hyper-parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embd = 384
embd_ffn = 4 * embd  # 4times as in "attention is all you need paper"
num_heads = 6  # every head is embd/num_head dimensional
n_layers = 6
dropout = 0.2  # 20% of operations are randomly masked at each forward/backward pass
n_components = 20  # number of gaussians in the MDN output layer
block_size = 129 # maximum context length (max number of strokes of the dataset here)

class MDN(nn.Module):
    """
    Mixture density network compatible with full covariance.
    Adapted from https://github.com/haimengzhao/full-cov-mdn

    [ Bishop, 1994 ]

    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    full_cov: bool; whether to use full or diagonal covariance matrix
    """
    def __init__(self, dim_in, dim_out, n_components, full_cov=True):
        super().__init__()
        self.pi_net = OneHotCategoricalNetwork(dim_in, n_components)
        self.normal_net = NormalNetwork(dim_in, dim_out, n_components, full_cov)

    def forward(self, x, tau=1.):
        return self.pi_net(x, tau), self.normal_net(x, tau)


class NormalNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, n_components, full_cov=True):
        super().__init__()
        self.n_components = n_components
        self.out_dim = out_dim
        self.full_cov = full_cov
        self.tril_indices = torch.tril_indices(row=out_dim, col=out_dim, offset=0)
        self.mean_net = nn.Linear(in_dim, out_dim * n_components)
        if full_cov:
            # Cholesky decomposition of the covariance matrix
            self.tril_net = nn.Linear(in_dim, int(out_dim * (out_dim + 1) / 2 * n_components))
        else:
            self.tril_net = nn.Linear(in_dim, out_dim * n_components)

    def forward(self, x, tau=1.):
        mean = self.mean_net(x).reshape(x.shape[0], x.shape[1], self.n_components, self.out_dim) # B, T, M, d
        if self.full_cov:
            tril_values = self.tril_net(x).reshape(x.shape[0], x.shape[1], self.n_components, -1) # B, T, M, (d**2+d)/2
            tril = torch.zeros(mean.shape[0], mean.shape[1], mean.shape[2], mean.shape[3], mean.shape[3]).to(x.device) # B, T, M, d, d
            tril[:, :, :, self.tril_indices[0], self.tril_indices[1]] = tril_values
            # use diag = exp(diag) to ensure stric positivity of diagonal elements
            tril.diagonal(dim1=-2, dim2=-1)[:] = tril.diagonal(dim1=-2, dim2=-1).exp()
        else:
            tril = self.tril_net(x).reshape(x.shape[0], x.shape[1], self.n_components, -1)
            tril = torch.diag_embed(tril.exp())
        tril *= tau
        return MultivariateNormal(mean, scale_tril=tril)

class OneHotCategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Linear(in_dim, out_dim)

    def forward(self, x, tau=1.):
        logits = self.network(x) / tau
        return OneHotCategorical(logits=logits)

class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.network = nn.Linear(in_dim, out_dim)

    def forward(self, x, tau=1.):
        logits = self.network(x) / tau
        return Categorical(logits=logits)


# Attention Head
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embd, head_size, bias=False)
        self.key = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # B, T, C
        k = self.key(x) # B, T, C

        # compute an attention score ("affinities")
        wei = q@k.transpose(-2, -1) * C **(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")) # "decoder" block because of triangular masking (autoregressive setting)
        wei = F.softmax(wei, dim=-1)

        # perform the weighted aggregation of the values
        v = self.value(x)  # B, T, C
        out = wei @ v # B, T, C

        return out

class MultiHead(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embd, embd) #projection layer going back into the residual pathway

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):

    def __init__(self, embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd, embd_ffn),
            nn.ReLU(),
            nn.Linear(embd_ffn, embd), # projection layer going back into the residual pathway
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer Block: communication/sensing followed by computation/update"""

    def __init__(self, embd, num_heads):
        super().__init__()
        self.sa_heads = MultiHead(num_heads, embd//num_heads)
        self.ffwd = FeedForward(embd)
        self.ln1 = nn.LayerNorm(embd) #should be equivalent to LayerNorm1D
        self.ln2 = nn.LayerNorm(embd)

    def forward(self, x):
        # x = self.sa_heads(x) # apply one head of self-attention (B, T, C) <=> "comunication" or "sense"
        # x = self.ffwd(x) # (B, T, C) => this is one a per-token level <=> "update"
        x = x + self.sa_heads(self.ln1(x)) # residual connection <=> "highway" of information and residual paths
        x = x + self.ffwd(self.ln2(x)) # residual connection

        return x

class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.stroke_embedding_proj = nn.Linear(2, embd, bias=False)
        self.pen_embedding_table = nn.Embedding(3, embd)
        self.position_embedding_table = nn.Embedding(block_size, embd)
        self.blocks = nn.Sequential(*[Block(embd, num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embd)
        self.mdn_head = MDN(embd, 2, n_components)
        self.pen_head = CategoricalNetwork(embd, 3)

    def forward(self, x, tau=1.):
        B, T, C = x.shape
        # assert C == 3

        # idx and targets are both (B,T) tensor of integers
        stroke_emb = self.stroke_embedding_proj(x[:, :, :2]) # (B,T,2) @ (2, embd) = (B, T, embd)
        pen_emb = self.pen_embedding_table(x[:, :, 2].long()) # (B, T, embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, embd)
        x = stroke_emb + pen_emb + pos_emb # (B, T, embd)

        # forward through attention heads
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)

        # forward though mdn and head
        pi_net, normal_net = self.mdn_head(x, tau=tau)
        q_net = self.pen_head(x, tau=tau)

        return pi_net, normal_net, q_net


    def loss(self, x, targets, mask):
        pi, normal, q = self.forward(x)
        ys = targets[:, :, :2]
        loglik = normal.log_prob(ys.unsqueeze(-2).expand_as(normal.loc))
        Ls = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=-1)
        Ls *= mask


        yp = targets[:, :, 2]
        Lp = -q.log_prob(yp)
        return Ls + Lp


    def sample(self, x, tau=1.):
        pi, normal, q = self.forward(x, tau)
        s_samples = torch.sum(pi.sample().unsqueeze(-1) * normal.sample(), dim=-2)
        p_samples = q.sample()
        return torch.cat([s_samples, p_samples.unsqueeze(-1)], dim=-1)


    @torch.no_grad()
    def generate(self, x, max_new_tokens, tau=1., break_eos=True):

        # x is (1, T, 3)
        for _ in range(max_new_tokens):

            # get the predictions
            samples_next = self.sample(x, tau=tau)[:, -1, :].unsqueeze(1)

            # append sampled stroke + pen index to the running sequence
            x = torch.cat([x, samples_next], dim=1)

            # break if end of sketch
            if break_eos:
                if samples_next[0,0,2] == 2:
                    return x

        return x

# helper function for draw_strokes
def get_bounds(data, factor):
  min_x = 0
  max_x = 0
  min_y = 0
  max_y = 0

  abs_x = 0
  abs_y = 0
  for i in range(len(data)):
    x = float(data[i,0])/factor
    y = float(data[i,1])/factor
    abs_x += x
    abs_y += y
    min_x = min(min_x, abs_x)
    min_y = min(min_y, abs_y)
    max_x = max(max_x, abs_x)
    max_y = max(max_y, abs_y)

  return (min_x, max_x, min_y, max_y)

def create_path(data, factor, abs_x, abs_y, lift_pen=1):
  command = "m"
  p = "M%s,%s " % (abs_x, abs_y)
  for i in range(len(data)):
    if (lift_pen == 1):
      command = "m"
    elif (command != "l"):
      command = "l"
    else:
      command = ""
    x = float(data[i,0])/factor
    abs_x += x
    y = float(data[i,1])/factor
    abs_y += y
    lift_pen = data[i, 2]
    p += command+str(x)+","+str(y)+" "
  return p, abs_x, abs_y

# little function that displays vector images
def draw_strokes(data, factor=0.2, svg_filename='sample.svg', the_color="black", stroke_width=1):
  min_x, max_x, min_y, max_y = get_bounds(data, factor)
  dims = (50 + max_x - min_x, 50 + max_y - min_y)
  dwg = svgwrite.Drawing(svg_filename, size=dims)
  dwg.add(dwg.rect(insert=(0, 0), size=dims,fill='white'))
  abs_x = 25 - min_x
  abs_y = 25 - min_y
  p, _, _ = create_path(data, factor, abs_x, abs_y)
  dwg.add(dwg.path(p).stroke(the_color,stroke_width).fill("none"))
  dwg.save()
  svg_str = dwg.tostring()
  return svg_str

if __name__ == "__main__":
    n_samples = 10
    tau = 0.4

    # Load Model
    model = TransformerModel()
    model = model.to(device)

    model_state_dict = torch.load('model_cat.pth', map_location=torch.device(device))
    model.load_state_dict(model_state_dict)
    model.eval()

    print(f"Loaded model has {sum([p.nelement() for p in model.parameters()])} parameters")

    # Generate Random Samples
    xs_start = torch.randn((n_samples, 1, 3), dtype=torch.float32)
    xs_start[:, :, 2] = 0.
    xs = [model.generate(x_start.unsqueeze(0), max_new_tokens=block_size-1, break_eos=True, tau=tau) for x_start in xs_start]

    for i, x in enumerate(xs):
        svg_str = draw_strokes(x[0], svg_filename=f"sample_{i}.svg", factor=0.1)