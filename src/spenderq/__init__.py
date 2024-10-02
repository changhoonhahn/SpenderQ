# -*- coding: utf-8 -*-
import torch
from torch import nn
from spender import SpectrumAutoencoder

from .desi_qso import DESI 


__all__ = [""]

__author__ = "ChangHoon Hahn"
__email__ = "changhoon.hahn@princeton.edu"
__uri__ = "https://github.com/changhoonhahn/SpenderQ"
__license__ = "MIT"
__copyright__ = "Copyright 2024 ChangHoon Hahn"


def load_model(filename, instruments=None, latents=10, z_min=2.1, z_max=3.5, bins=9780,
        n_hidden=(64, 256, 1024)):
    ''' load spenderq model from file 
    '''
    if instruments is None: 
        instruments = [DESI()]

    # restframe wavelength for reconstructed spectra
    # Note: represents joint dataset wavelength range
    lmbda_min = instruments[0].wave_obs[0]/(1.0+z_max) # 2000 A
    lmbda_max = instruments[0].wave_obs[-1]/(1.0+z_min) # 9824 A
    wave_rest = torch.linspace(lmbda_min, lmbda_max, bins, dtype=torch.float32)

    # define and train the model
    models = [ SpectrumAutoencoder(instrument,
                                   wave_rest,
                                   n_latent=latents,
                                   n_hidden=n_hidden,
                                   act=[nn.LeakyReLU()]*(len(n_hidden)+1)
                                   )
              for instrument in instruments ]

    device = instruments[0].wave_obs.device
    model_struct = torch.load(filename, map_location=device)
    
    for i, model in enumerate(models):
        # backwards compat: encoder.mlp instead of encoder.mlp.mlp
        if 'encoder.mlp.mlp.0.weight' in model_struct['model'][i].keys():
            from collections import OrderedDict
            model_struct['model'][i] = OrderedDict([(k.replace('mlp.mlp', 'mlp'), v) for k, v in model_struct['model'][i].items()])
        # backwards compat: add instrument to encoder
        try:
            model.load_state_dict(model_struct['model'][i], strict=False)
        except RuntimeError:
            model_struct['model'][i]['encoder.instrument.wave_obs']= instruments[i].wave_obs
            #model_struct['model'][i]['encoder.instrument.skyline_mask']= instruments[i].skyline_mask
            model.load_state_dict(model_struct[i]['model'], strict=False)

    losses = model_struct['losses']
    return models, losses
