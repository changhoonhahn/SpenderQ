<img src="./doc/spenderq_logo.png" width="400">
Spectrum autoencoder framework for reconstructing quasar spectra and measuring the Ly$\alpha$ forest. 
<br>
<br>

Quasar spectra carry the imprint of foreground intergalactic medium (IGM) through absorption systems. In particular, absorption caused by neutral hydrogen gas, the "Ly$\alpha$ forest," is a key spectroscopic tracer for cosmological analyses used to measure cosmic expansion and test physics beyond the standard model. `SpenderQ` is a ML-based  framework for reconstructing intrinsic quasar spectra and measuring the Lyα forest from observations. `SpenderQ` uses the [`Spender` spectrum autoencoder](https://github.com/pmelchior/spender) to learn a compact and redshift-invariant latent encoding of quasar spectra, combined with an iterative procedure to identify and mask absorption regions. It is entirely data-driven (e.g., not calibrated on simulations) and makes no assumptions on the shape of the intrinsic quasar continuum. 

Here's a schematic diagram of the `SpenderQ` framework:

<img src="./doc/spenderq_framework.png" width="600">

and an example on a mock spectrum (grey/black) where we know the truth (blue): 

<img src="./doc/spenderq_demo.png" width="800">

Here's `SpenderQ` in action on real public DESI data: 
<img src="./doc/spenderq_demo.gif" width="800">

Here's another for a spectrum with a Broad Absoprtion Line, which was not
masked
<img src="./doc/spenderq_demo_bal.gif" width="800">




## Team 

ChangHoon Hahn (Princeton; changhoon.hahn[at]princeton.edu)

Satya Gontcho A Gontcho (Berkeley)

Peter Melchior (Princeton)

Abby Bault (Princeton)

Hiram Herrera (CEA)
