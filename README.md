# LatentOps [WIP]
Source code of LatentOps.

Paper: ***Composable Text Control Operations in Latent Space with Ordinary Differential Equations***

*https://arxiv.org/abs/2208.00638*

## Recommended Environment
We recommend to create a new conda enviroment (named *latentops*) by:
```
$ conda create -n latentops python==3.9.1 pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```
Then activate *latentops* and install the required packages by running:
```
$ conda activate latentops
$ bash build_envs.sh
```


## Prepare Datasets
There are two ways to download the datasets (the second way is recommonded)

 - Download from Onedrive and unzip into [*./data*](/data) folder : [Link](https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/ETzJ0Fae4-lHi3vN8G8HYbQBvZr7wh7iQvqMCd2YloAb_g?e=8CpDkl)
 - Download and process the datasets by running the scripts:
 
    ```
    $ bash download_datasets.sh
    ```

## Pretrained Models
There are two ways to download the pretrained models (the second way is recommonded)

 - Download from Onedrive and unzip into [*./ckpts*](/ckpts) folder : [Link](https://cuhko365-my.sharepoint.com/:f:/g/personal/218019026_link_cuhk_edu_cn/ElZdkwSkQtRKrJ94Eh-KMAIBJfm2cwUoBVI0TbwIik06Wg?e=mWSVAj)
 - Download and process the pretrained model by running the scripts:
 
    ```
    $ bash download_pretrained_models.sh
    ```
    

  
    
## Outputs
To facilitate comparision, we provide the output files of text style transfer in [*./outputs*](/outputs) folder.



## Cite
```
@article{liu2022composable,
    title={Composable Text Control Operations in Latent Space with Ordinary Differential Equations},
    author={Liu, Guangyi and Feng, Zeyu and Gao, Yuan and Yang, Zichao and Liang, Xiaodan and Bao, Junwei and He, Xiaodong and Cui, Shuguang and Li, Zhen and Hu, Zhiting},
    journal={arXiv preprint arXiv:2208.00638},
    year={2022}
}
```
***Code is coming soon...***
