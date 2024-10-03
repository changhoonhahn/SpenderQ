#!/usr/bin/env python
import argparse
import os

import pickle 
import numpy as np
import torch

from spenderq import util, load_model
from spenderq import lyalpha as LyA


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="data file directory")
    parser.add_argument("model", help="filename of model")
    parser.add_argument("-ti", "--input_tag", help="input data tag", type=str, default='')
    parser.add_argument("-to", "--output_tag", help="output data tag", type=str, default='')
    parser.add_argument("-i", "--ibatch", help="batch number", type=int, default=None)
    parser.add_argument("-sigma", "--sigma", help="sigma, for clipping", type=float, default=1.5)
    args = parser.parse_args()
    
    # load model 
    models, losses = load_model(args.model) 

    # load data  
    fbatch = os.path.join(args.dir, '%s_%i.pkl' % (args.input_tag, args.ibatch))
    with open(fbatch, "rb") as f:
        spec, w, z, target_id, norm, zerr = pickle.load(f)
    
    with torch.no_grad():
        models[0].eval()

        s = models[0].encode(spec)
        recon = models[0].decode(s)
        
    for igal in np.arange(spec.shape[0]): 
        # identify LyA absorption
        is_absorb = LyA.identify_LyA(
            np.array(models[0].wave_obs), 
            np.array(spec[igal]), 
            np.array(w[igal]), 
            np.array(z)[igal], 
            np.array(models[0].wave_rest * (1 + z[igal])), 
            np.array(recon[igal]), sigma=args.sigma)
        
        # update weights
        w[igal,is_absorb] = 0.
    
    # save updated batch 
    batch = [spec, w, z, target_id, norm, zerr]
    fbatch = os.path.join(args.dir, '%s_%i.pkl' % (args.output_tag, args.ibatch))
    with open(fbatch, "wb") as f:
        pickle.dump(batch, f)

    # save latents and reconstruction
    flatent = os.path.join(args.dir, '%s_%i.latents.npy' % (args.input_tag, args.ibatch))
    np.save(flatent, np.array(s)) 
    
    frecon = os.path.join(args.dir, '%s_%i.recons.npy' % (args.input_tag, args.ibatch))
    np.save(frecon, np.array(recon)) 
