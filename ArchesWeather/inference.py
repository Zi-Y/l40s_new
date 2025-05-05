from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import torch

torch.set_grad_enabled(False)

# load model and dataset
device = 'cuda:0'
cfg = OmegaConf.load('modelstore/archesweather-M/archesweather-M_config.yaml')

ds = instantiate(cfg.dataloader.dataset,
                    path='data/era5_240/full/',
                    domain='test') # the test domain is year 2020

backbone = instantiate(cfg.module.backbone)
module = instantiate(cfg.module.module, backbone=backbone, dataset=ds)

ckpt = torch.load('modelstore/archesweather-M/archesweather-M_weights.pt', map_location='cpu')
module.load_state_dict(ckpt)
module = module.to(device).eval()


# make a batch
batch = {k:(v[None].to(device) if hasattr(v, 'to') else [v]) for k, v in ds[0].items()}
output = module.forward(batch)

# denormalize output
denorm_pred = ds.denormalize(output, batch)

# get per-sample main metrics from WeatherBench
from evaluation.deterministic_metrics import headline_wrmse
denorm_batch = ds.denormalize(batch)
metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')

# average metrics
metrics_mean = {k:v.mean(0) for k, v in metrics.items()}

#plot prediction
plt.imshow(denorm_pred['next_state_surface'][0, 2, 0].detach().cpu().numpy())
plt.imsave('results.jpg', denorm_pred['next_state_surface'][0, 2, 0].detach().cpu().numpy())
