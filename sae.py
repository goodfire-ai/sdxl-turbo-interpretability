import torch
from math import prod

class BatchTopKSAE(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        k,
        device,
        dtype,
        tiebreaker_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        
        # Initialize encoder and decoder weights
        encoder_mat = torch.randn((d_in, d_hidden))
        encoder_mat = 0.1 * encoder_mat / torch.linalg.norm(encoder_mat, dim=0, ord=2, keepdim=True)
        decoder_mat = torch.randn((d_hidden, d_in))
        decoder_mat = 0.1 * decoder_mat / torch.linalg.norm(decoder_mat, dim=1, ord=2, keepdim=True)
        
        # Create separate linear layers for encoder and decoder
        self.encoder = torch.nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_in, bias=True)
        
        # Initialize the weights and biases
        self.encoder.weight.data = encoder_mat.T  # Transpose for nn.Linear convention
        self.encoder.bias.data = torch.zeros(self.d_hidden)
        self.decoder.weight.data = decoder_mat.T  # Transpose for nn.Linear convention
        self.decoder.bias.data = torch.zeros(self.d_in)
        
        self.device = device
        self.dtype = dtype
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(self.device, self.dtype)
        
    def encoder_pre(self, x):
        return self.encoder(x)

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)
    
    def _batch_topk(self, f, k, tiebreak=False):
        if tiebreak:  # break ties in feature order for determinism
            f = f + self.tiebreaker.to(f.device, f.dtype).broadcast_to(f.shape)
        *input_shape, _ = f.shape  # handle higher-dim tensors (e.g. from sequence input)
        numel = k * prod(input_shape)
        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        f_topk = torch.zeros_like(f.flatten()).scatter(-1, f_topk.indices, f_topk.values).reshape(f.shape)
        return f_topk

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f


def load_sdxl_turbo_sae(file_path, device, dtype):
    state_dict = torch.load(file_path, weights_only=True, map_location=device)
    sae = BatchTopKSAE(d_in=1280, d_hidden=1280 * 8, k=10, dtype=dtype, device=device)
    sae.load_state_dict(state_dict)
    return sae