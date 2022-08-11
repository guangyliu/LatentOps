# LatentOps [WIP]
Source code of paper: ***Composable Text Control Operations in Latent Space with Ordinary Differential Equations***

*https://arxiv.org/abs/2208.00638*


***Code is coming soon...***

The code is based on [*Optimus*](https://github.com/ChunyuanLI/Optimus) and [*LACE*](https://github.com/NVlabs/LACE).
## Preparation
### Recommended Environment
We recommend to create a new conda enviroment (named *latentops*) by:
```shell
conda create -n latentops python==3.9.1 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Then activate *latentops* and install the required packages by running:
```shell
conda activate latentops
bash build_envs.sh
```

### Prepare Datasets
Download and process the datasets by running the script:
```shell
bash download_datasets.sh
```

### Pretrained Models
Download and process the pretrained model by running the script:
```shell
bash download_pretrained_models.sh
```
    
### Prepare Classifiers
Download and process the classifiers by running the script:
 ```shell
 bash download_classifiers.sh
 ```
    
## Outputs
To facilitate comparison, we provide the output files of text style transfer in [*./outputs*](/outputs) folder.



## Cite
```
@article{liu2022composable,
    title={Composable Text Control Operations in Latent Space with Ordinary Differential Equations},
    author={Liu, Guangyi and Feng, Zeyu and Gao, Yuan and Yang, Zichao and Liang, Xiaodan and Bao, Junwei and He, Xiaodong and Cui, Shuguang and Li, Zhen and Hu, Zhiting},
    journal={arXiv preprint arXiv:2208.00638},
    year={2022}
}
```

