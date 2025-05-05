import pytorch_lightning as pl
import torch.nn as nn
import torch
import diffusers
from pathlib import Path
import numpy as np
import torch.utils.checkpoint as gradient_checkpoint
from evaluation.deterministic_metrics import headline_wrmse
import os
from torch.optim.lr_scheduler import LambdaLR

lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/120)])
lat_coeffs_equi =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]

pressure_levels = torch.tensor([  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
       1000]).float()
level_coeffs = (pressure_levels/pressure_levels.mean())[None, None, :, None, None]
graphcast_surface_coeffs = torch.tensor([0.1, 0.1, 1.0, 0.1])[None, :, None, None, None] # graphcast
pangu_surface_coeffs = torch.tensor([0.25, 0.25, 0.25, 0.25])[None, :, None, None, None] # pangu coeffs

from copy import deepcopy

class ForecastModule(pl.LightningModule):
    def __init__(self, 
                 backbone,
                 dataset=None,
                 samples_loss_np=0,
                 save_per_sample_loss=False,
                 path_save_base='./',
                 use_info_batch=False,
                 pow=2, # 2 is standard mse
                 lr=1e-4, 
                 betas=(0.9, 0.98),
                 weight_decay=1e-5,
                 num_warmup_steps=1000, 
                 num_training_steps=300000,
                 num_cycles=0.5,
                 use_graphcast_coeffs=False,
                 decreasing_pressure=False,
                 increase_multistep_period=2,
                 **kwargs
                ):
        ''' should create self.encoder and self.decoder in subclasses
        '''
        super().__init__()
        #self.save_hyperparameters()
        self.__dict__.update(locals())
        self.backbone = backbone # necessary to put it on device
        #self.area_weights = dataset.area_weights[None, None, None]
        self.area_weights = lat_coeffs_equi
        # if hasattr(dataset, 'area_weights'):
        #     self.area_weights = dataset.area_weights[None, None, None]
        # else:
        #     self.area_weights = dict(equi=lat_coeffs_equi, 
        #                          cubed=lat_coeffs_cubed)[mode]

        '''
        first 4 columns are for surface variables
        the next 13*6 columns are for upper-air variables
        the same variable first,
        variable1_level1, variable1_level2, ..., variable1_leve13,
        variable2_level1, variable2_level2, ..., variable6_level13
        the last columns is for total weighted loss
        '''
        self.samples_loss_np = samples_loss_np
        self.save_per_sample_loss = save_per_sample_loss
        self.use_info_batch = use_info_batch
        self.line_index_info_batch = 0

        self.indicator_info_batch_stop = False

        # if isinstance(samples_loss_np, np.ndarray):
        #     self.save_per_sample_loss = True
        # else:
        #     self.save_per_sample_loss = False

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        
    def forward(self, batch):
        input_surface = torch.cat([batch['state_surface'],
                                     batch['state_constant']], dim=1)
        input_surface = input_surface.squeeze(-3)
        if self.decreasing_pressure:
            input_level = batch['state_level'].flip(-3)
        else:
            input_level = batch['state_level']
                                     
        out = self.backbone(input_surface, input_level)
        return out

    def forward_multistep(self, batch):
        # multistep forward with gradient checkpointing to save GPU memory
        lead_iter = int((batch['lead_time_hours'][0]//24).item())
        preds_future = []
        input_batch = batch
        denorm_input_batch = self.dataset.denormalize(input_batch)

        pred = self.forward(batch)
        next = pred # save first prediction
        
        for i in range(lead_iter-1):
            batch = self.dataset.normalize_next_batch(pred, batch)
            pred = gradient_checkpoint.checkpoint(self.forward, batch, use_reentrant=False)

            denorm_pred = self.dataset.denormalize(pred, batch)

            # renormalize with state from input_batch and save to preds_future
            denorm_pred['state_level'] = denorm_input_batch['state_level']
            denorm_pred['state_surface'] = denorm_input_batch['state_surface']
            renorm_pred = self.dataset.normalize(denorm_pred)

            preds_future.append(renorm_pred)

        future = dict(future_state_level=torch.stack([state_future['next_state_level']
                                                      for state_future in preds_future], dim=1),
                    future_state_surface=torch.stack([state_future['next_state_surface']
                                                      for state_future in preds_future], dim=1))
        return next, future
            
    def mylog(self, dct={}, **kwargs):
        #print(mode, kwargs)
        mode = 'train_' if self.training else 'val_'
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode+k, v, prog_bar=True, sync_dist=True, add_dataloader_idx=True)
            
    def loss(self, pred, batch, prefix='next_state', multistep=False, **kwargs):
        device = batch['next_state_level'].device
        mse_surface = (pred[prefix+'_surface'] - batch[prefix+'_surface']).abs().pow(self.pow)
        mse_surface = mse_surface.mul(self.area_weights.to(device)) # latitude coeffs

        # if self.use_info_batch:
        # sample_id = (self.trainer.global_step % 58440)
        sample_id = self.line_index_info_batch
        self.line_index_info_batch += 1
        if self.line_index_info_batch >= (self.samples_loss_np.shape[0]):
            self.line_index_info_batch = 0
        # else:
        #     sample_id = batch['id'].item()
        if self.training and self.save_per_sample_loss:
            mse_surface_saved = deepcopy(mse_surface.detach().cpu())
            mse_surface_saved = torch.squeeze(mse_surface_saved)
            mse_surface_saved = torch.mean(mse_surface_saved.reshape(4, -1), dim=1)
            self.samples_loss_np[sample_id][0:4] = mse_surface_saved.numpy()
            # print('id {}, gpu_id {}, loss {}'.format(sample_id, mse_surface.device.index,
            #                                          self.samples_loss_np[sample_id][0]))


        surface_coeffs = pangu_surface_coeffs if not self.use_graphcast_coeffs else graphcast_surface_coeffs
        mse_surface_w = mse_surface.mul(surface_coeffs.to(device))


        mse_level = (pred[prefix+'_level'] - batch[prefix+'_level']).pow(self.pow)
        mse_level = mse_level.mul(self.area_weights.to(device))
        # test_only
        if self.training and self.save_per_sample_loss:
            mse_level_saved = deepcopy(mse_level.detach().cpu())
            mse_level_saved = torch.squeeze(mse_level_saved)
            mse_level_saved = torch.mean(mse_level_saved.reshape(6*13, -1), dim=1)
            # if self.use_info_batch:
            self.samples_loss_np[sample_id][4:-4] = mse_level_saved.numpy()
            self.samples_loss_np[sample_id][-4] = batch['id'].item()
            self.samples_loss_np[sample_id][-3] += 1

            # else:
            #     self.samples_loss_np[sample_id][4:-2] = mse_level_saved.numpy()


        mse_level_w = mse_level.mul(level_coeffs.to(device))
    

        nvar_level = mse_level_w.shape[-4]
        nvar_surface = surface_coeffs.sum().item()

        if multistep:
            lead_iter = int((batch['lead_time_hours'][0]//24).item())
            future_coeffs = torch.tensor([1/i**2 for i in range(2, lead_iter + 1)]).to(device)[None, :, None, None, None, None]
            mse_surface_w = mse_surface_w.mul(future_coeffs)
            mse_level_w = mse_level_w.mul(future_coeffs)

        
        # coeffs are for number of variables
        loss = (4*mse_surface_w.mean() + nvar_level*mse_level_w.mean())/(nvar_level + nvar_surface)
        if self.training and self.save_per_sample_loss:
            self.samples_loss_np[sample_id][-1] = loss.item()
            self.samples_loss_np[sample_id][-2] = self.trainer.global_step

        return mse_surface, mse_level, loss
        

    def training_step(self, batch, batch_nb):

        if not 'future_state_level' in batch:
            # standard prediction 
            pred = self.forward(batch)
            _, _, loss = self.loss(pred, batch, prefix='next_state')
            self.mylog(loss=loss)

            denorm_pred = self.dataset.denormalize(pred, batch)
            denorm_batch = self.dataset.denormalize(batch)

            metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
            metrics_mean = {k:v.mean(0) for k, v in metrics.items()} # average on time and pred delta

            self.mylog(**metrics_mean)

        else:
            # multistep prediction
            next, future = self.forward_multistep(batch)
            _, _, next_loss = self.loss(next, batch, prefix='next_state')
            _, _, future_loss = self.loss(future, batch, prefix='future_state', multistep=True)


            lead_iter = int((batch['lead_time_hours'][0]//24).item())
            self.mylog(lead_iter=lead_iter)

            loss = (next_loss + (lead_iter - 1)*future_loss)/lead_iter
            self.mylog(future_loss=future_loss)
            self.mylog(next_loss=next_loss)
            self.mylog(loss=loss)

            # log metrics for next
            denorm_pred = self.dataset.denormalize(next, batch)
            denorm_batch = self.dataset.denormalize(batch)

            metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
            metrics_mean = {k:v.mean(0) for k, v in metrics.items()}
            self.mylog(**metrics_mean)

            #log some metrics for second step only
            denorm_pred = self.dataset.denormalize(future, batch)
            denorm_batch = self.dataset.denormalize(batch)

            metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='future_state')
            metrics_mean = {k:v.mean(0)[0] for k, v in metrics.items()} # select the 48h lead time prediction with [0]
            self.mylog(**metrics_mean)

        if self.use_info_batch:
            loss = self.dataset.update(loss, batch['id'])

        # aaa = batch['id'].item()
        # print(f'batch_nb {batch_nb}, real ID {aaa}')
        if self.use_info_batch:
            if batch_nb >= (self.trainer.train_dataloader.dataset.dataset_len/len(self.trainer.device_ids)-1):
                self.trainer.should_stop = True
                self.indicator_info_batch_stop = True
                print(f'batch_nb {batch_nb}, training_step self.trainer.should_stop {self.trainer.should_stop}')

        return loss
        
        
    def validation_step(self, batch, batch_nb):
        # print('current epoch', self.current_epoch)
        pred = self.forward(batch)
        _, _, loss = self.loss(pred, batch)
        self.mylog(loss=loss)
        # denorm and compute metrics

        denorm_pred = self.dataset.denormalize(pred, batch)
        denorm_batch = self.dataset.denormalize(batch)

        metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
        metrics_mean = {k:v.mean(0) for k, v in metrics.items()} # average on time and pred delta

        self.mylog(**metrics_mean)
        return loss
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def on_train_epoch_start(self, outputs=None):
        # print('\n')
        # self.trainer.train_dataloader
        # self.trainer.train_dataloader.sampler = self.trainer.train_dataloader.dataset.sampler
        # print('Forecast - dataset len:', self.dataset.__len__())

        if self.dataset.multistep > 1:
            # increase multistep every 2 epochs
            self.dataset.multistep = 2 + self.current_epoch // self.increase_multistep_period
        # if (self.current_epoch > 0) and (self.current_epoch < self.dataset.stop_prune):
        #     self.trainer.limit_train_batches = self.dataset.dataset_len/len(self.dataset)
        #     print(f'on_train_epoch_start: dataset_len:{self.dataset.dataset_len}, total: {len(self.dataset)}')
        #     print('self.trainer.limit_train_batches:', self.trainer.limit_train_batches)
        # else:
        #     self.trainer.limit_train_batches = 1.0



    def on_train_epoch_end(self):

        # print('gpu_id {}, loss {}'.format(self.device.index,
        #                                          self.samples_loss_np[sample_id][0]))
        # test_only
        print('self.trainer.global_step', self.trainer.global_step)
        if self.indicator_info_batch_stop and self.use_info_batch:
            self.indicator_info_batch_stop = False
            self.trainer.should_stop = False
            print(f'on_train_epoch_end self.trainer.should_stop {self.trainer.should_stop}')

        if self.save_per_sample_loss:
            # np.save(self.samples_loss_np, "Epoch_{:0>3d}.np".format(self.current_epoch))
            # path = self.dataset.files[0]

            # save_path = ("/mnt/cache/data/zi/era5_240/full/loss_per_sample/")
            # save_path = ("/home/zi/research_project/loss_per_sample/3x3090_seed0_new/")

            # if self.use_info_batch:
            save_path = self.path_save_base + '/infobatch_loss_values/'
            # else:
            #     save_path = self.path_save_base + '/loss_values/'
            os.makedirs(save_path, exist_ok=True)

            # if self.use_info_batch:
            loss_epoch_np_cache = np.copy(self.samples_loss_np)
            zero_rows = np.all(loss_epoch_np_cache == 0, axis=1)
            loss_epoch_np_cache = loss_epoch_np_cache[~zero_rows]

            save_path = "{}Epoch_{}_device_{}.npy".format(save_path, self.current_epoch, self.device.index)
            np.save(save_path, loss_epoch_np_cache)
            # else:
            #     save_path = "{}Epoch_{}_device_{}.npy".format(save_path, self.current_epoch, self.device.index)
            #     np.save(save_path, self.samples_loss_np)

            self.samples_loss_np[:] = 0

    def on_train_end(self):
        dataloaders = self.trainer.val_dataloaders
        self.trainer.validate(model=self, dataloaders=dataloaders)


    def configure_optimizers(self):
        # def lr_lambda(current_step):
        #     decay_factor = 0.19
        #     step_interval = 50000
        #     num_warmup_steps = 5000
        #     if current_step < num_warmup_steps:
        #         # Warmup phase: linear increase
        #         return current_step / num_warmup_steps
        #     else:
        #         # Step decay phase
        #         num_steps_after_warmup = current_step - num_warmup_steps
        #         decay_steps = num_steps_after_warmup // step_interval
        #         return 1 - min((decay_steps * decay_factor), 1)

        print('configure optimizers')
        decay_params = {k: True for k, v in self.named_parameters() if 'weight' in k and not 'norm' in k}
        opt = torch.optim.AdamW([{'params': [v for k, v in self.named_parameters() if k in decay_params]},
                                 {'params': [v for k, v in self.named_parameters() if k not in decay_params],
                                  'weight_decay': 0}],
                                lr=self.lr,
                                betas=self.betas,
                                weight_decay=self.weight_decay)

        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles)

        # sched = LambdaLR(opt, lr_lambda=lr_lambda)

        sched = {'scheduler': sched,
                 'interval': 'step',  # or 'epoch'
                 'frequency': 1}
        return [opt], [sched]

    
class ForecastModuleWithCond(ForecastModule):
    '''
    module that can take additional information:
    - month and hour
    - previous state
    - pred state (e.g. prediction of other weather model)
    '''
    def __init__(self, *args, cond_dim=32, use_pred=False, use_prev=False, **kwargs):
        from backbones.dit import TimestepEmbedder
        super().__init__(*args, **kwargs)
        # cond_dim should be given as arg to the backbone
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.use_pred = use_pred
        self.use_prev = use_prev

    def forward(self, batch):
        device = batch['state_surface'].device
        input_surface = torch.cat([batch['state_surface'], 
                                   batch['state_constant']], dim=1)
        input_level = batch['state_level']
        if self.use_pred and 'pred_state_surface' in batch:
            input_surface = torch.cat([input_surface, batch['pred_state_surface']], dim=1)
            input_level = torch.cat([input_level, batch['pred_state_level']], dim=1)

        if self.use_prev and 'prev_state_surface' in batch:
            input_surface = torch.cat([input_surface, batch['prev_state_surface']], dim=1)
            input_level = torch.cat([input_level, batch['prev_state_level']], dim=1)
            
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor([int(x[-5:-3]) for x in batch['time']]).to(device)
        hour_emb = self.hour_embedder(hour)

        t_emb = month_emb + hour_emb

        input_surface = input_surface.squeeze(-3)
        out = self.backbone(input_surface, input_level, t_emb)
        return out
        
        
