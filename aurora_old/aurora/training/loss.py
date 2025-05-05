from datetime import datetime
import torch
import torch.optim as optim
import torch.utils.checkpoint as checkpoint


class AuroraMeanAbsoluteError(torch.nn.Module):

    def __init__(
        self,
        alpha=1 / 4,
        beta=1,
        variable_weights=None,
        dataset_weights=None,
    ):
        super(AuroraMeanAbsoluteError, self).__init__()

        self.surf_var_names = tuple(variable_weights["surface"].keys())
        self.atmos_var_names = tuple(variable_weights["atmospheric"].keys())

        # define weights tensor
        self.surf_weights = torch.tensor([variable_weights["surface"][name] for name in self.surf_var_names])
        self.atmos_weights = torch.tensor([variable_weights["atmospheric"][name] for name in self.atmos_var_names])

        self.alpha = alpha
        self.beta = beta
        self.dataset_weights = dataset_weights if dataset_weights else {"default": 1.0}


    def forward(self, pred, target, dataset_name="default"):
        gamma = self.dataset_weights.get(dataset_name, 1.0)

        # Surface loss calculation
        pred_surf = torch.stack([pred.surf_vars[var] for var in self.surf_var_names]).squeeze()
        target_surf = torch.stack([target.surf_vars[var] for var in self.surf_var_names]).squeeze()
        # surf_diffs is avg. over H × W
        # pred.metadata.lat.detach().cpu().numpy()
        # pred.metadata.lon.detach().cpu().numpy()
        surf_diffs = torch.abs(pred_surf - target_surf).mean(dim=(1, 2))
        surface_loss = (surf_diffs * self.surf_weights.to(surf_diffs.device)).sum()
        surface_loss *= self.alpha

        # Atmospheric loss calculation
        pred_atmos = torch.stack([pred.atmos_vars[var] for var in self.atmos_var_names]).squeeze()
        target_atmos = torch.stack([target.atmos_vars[var] for var in self.atmos_var_names]).squeeze()

        # atmos_diffs is avg. over C × H × W
        atmos_diffs = torch.abs(pred_atmos - target_atmos).mean(dim=(1, 2, 3))
        atmospheric_loss = (atmos_diffs * self.atmos_weights.to(atmos_diffs.device)).sum()
        atmospheric_loss *= self.beta

        # Total loss calculation
        total_vars = len(self.surf_var_names) + len(self.atmos_var_names)
        total_loss = (gamma / total_vars) * (surface_loss + atmospheric_loss)

        return total_loss
