#!/bin/bash
# The original SharePoint archive is no longer available. The same data is rebuilt
# from the public Li et al. (2018) corpora:
python data/prepare_data.py --datasets yelp amazon "$@"
echo "Datasets are in ./data/datasets"
