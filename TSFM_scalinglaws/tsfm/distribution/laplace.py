from typing import Callable, Optional

import torch
from jaxtyping import Float, PyTree
from torch.distributions import Laplace
from torch.nn import functional as F

from ._base import DistributionOutput


class LaplaceOutput(DistributionOutput):
    distr_cls = Laplace
    args_dim = dict(loc=1, scale=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(loc=self._loc, scale=self._scale)

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)


class LaplaceFixedScaleOutput(DistributionOutput):
    distr_cls = Laplace
    args_dim = dict(loc=1)

    def __init__(self, scale: float = 1e-3):
        self.scale = scale

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(loc=self._loc)

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    def _distribution(
        self,
        distr_params: PyTree[Float[torch.Tensor, "*batch 1"], "T"],
        validate_args: Optional[bool] = None,
    ) -> Laplace:
        loc = distr_params["loc"]
        distr_params["scale"] = torch.as_tensor(
            self.scale, dtype=loc.dtype, device=loc.device
        )
        return self.distr_cls(**distr_params, validate_args=validate_args)
