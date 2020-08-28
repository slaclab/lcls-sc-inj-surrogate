[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jacquelinegarrahan/lcls-sc-inj-surrogate/master?urlpath=/proxy/5006/surrogate_model_client)

# lcls_sc_inj_surrogate
Repository for LCLS Superconducting Injector Surrogate Modeling

This surrogate model was trained using Astra simulations from the following data types:

1. Astra simulations using various laser inputs types.
    * Particle distributions generated using [distgen](https://github.com/ColwynGulliford/distgen)

2. Measured data collected at SLAC in October and November 2019
    * Image processing scripts which replicate the procedure applied to the measured data, before training, are included.

## Navigating this repository

1. PV information for measured data, as well as measured data used for training (pre processed and processed) are available. 
2. Current models, compatible with [lume-model](https://github.com/jacquelinegarrahan/lume-model) and [lume-epics](https://github.com/slaclab/lume-epics). 


### Support

For further information, please contact [Lipi Gupta](lipigupta@uchicago.edu) or [Chris Mayes](cmayes@stanford.edu) for more information.
