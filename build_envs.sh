#!/bin/bash
# Install the pinned Python dependencies. Run inside the `latentops` conda env
# created as described in README.md.
set -e
pip install -r requirements.txt

# NVIDIA Apex is OPTIONAL. It is only used when the training scripts pass
# --fp16 (apex_opt=O2). Skip it if the build fails on your CUDA version.
if [ "$1" == "--with-apex" ]; then
  git clone https://github.com/NVIDIA/apex
  cd apex && pip install -v --disable-pip-version-check --no-cache-dir ./ && cd ..
fi
