# legacy/

Exploratory code that is **not** on the path used by the EMNLP 2023 paper and is not
maintained: alternative samplers and transfer scripts (`lace_*`), a latent-diffusion
variant (`ddpm*`, `train_ddpm_latent*`), SDE experiments (`test_SBM*`), SimCSE / PPVAE /
DAAE / ARAE / CARA / SpaceFusion baselines, and older VAE-training variants.

Nothing in `code/examples/big_ae/` imports from here. The files are kept so that old
commands still resolve, but expect hard-coded paths and missing checkpoints.
