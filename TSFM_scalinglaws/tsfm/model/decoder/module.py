from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from tsfm.common.torch_util import packed_attention_mask
from tsfm.distribution import DistributionOutput
from tsfm.module.norm import RMSNorm
from tsfm.module.packed_scaler import PackedNOPScaler, PackedStdScaler
from tsfm.module.position import (
    QueryKeyProjection,
    RotaryProjection,
)
from tsfm.module.transformer import TransformerEncoder


def encode_distr_output(
    distr_output: DistributionOutput,
) -> dict[str, str | float | int]:
    def _encode(val):
        if not isinstance(val, DistributionOutput):
            return val

        return {
            "_target_": f"{val.__class__.__module__}.{val.__class__.__name__}",
            **tree_map(_encode, val.__dict__),
        }

    return _encode(distr_output)


def decode_distr_output(config: dict[str, str | float | int]) -> DistributionOutput:
    return instantiate(config, _convert_="all")


class BasicModule(
    nn.Module,
    PyTorchModelHubMixin,
    coders={DistributionOutput: (encode_distr_output, decode_distr_output)},
):
    """Contains components of Moirai to ensure implementation is identical across models"""

    def __init__(
        self,
        distr_output: DistributionOutput,
        d_model: int,
        num_layers: int,
        patch_size: int,  # tuple[int, ...] | list[int]
        max_seq_len: int,
        attn_dropout_p: float,
        dropout_p: float,
        num_heads: int = None,
        scaling: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.scaling = scaling

        self.scaler = PackedStdScaler() if scaling else PackedNOPScaler()
        self.in_proj = nn.Linear(patch_size, d_model)
        self.encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=num_heads,
            pre_norm=True,
            attn_dropout_p=attn_dropout_p,
            dropout_p=dropout_p,
            norm_layer=RMSNorm,
            activation=F.silu,
            use_glu=True,
            use_qk_norm=True,
            var_attn_bias_layer=None,
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
                kwargs=dict(max_len=max_seq_len),
                partial_factor=(0.0, 0.5),
            ),
            shared_var_attn_bias=False,
            shared_time_qk_proj=True,
            d_ff=None,
        )
        self.distr_output = distr_output
        self.param_proj = self.distr_output.get_param_proj(d_model, patch_size)
    
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> Distribution:
        loc, scale = self.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
        scaled_target = (target - loc) / scale
        reprs = self.in_proj(scaled_target)
        
        attn_mask = packed_attention_mask(sample_id)
        attn_mask = attn_mask & torch.ones_like(attn_mask).tril(diagonal=0)
        
        reprs = self.encoder(
            reprs,
            attn_mask,
            time_id=time_id,
            var_id=variate_id,
        )
        
        distr_param = self.param_proj(reprs)
        distr = self.distr_output.distribution(distr_param, loc=loc, scale=scale)
        return distr
        