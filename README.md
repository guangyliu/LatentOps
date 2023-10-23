# LatentOps [WIP]
Source code of paper: ***Composable Text Controls in Latent Space with ODEs***

*https://arxiv.org/abs/2208.00638*


***Code is coming soon...***

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
Download and process the external classifiers by running the script:
 ```shell
 bash download_classifiers.sh
 ```
## Conditional Generation
You can do conditional generation (default Yelp) by running:
```shell
cd code
bash conditional_generation.sh $1 $2
```
$1 represents operators (1 for sentiment, 4 for tense, 33 for formality).
$2 represents desired labels:
- sentiment: 0-negative, 1-positive
- tense: 0-past, 1-present, 2-future
- formality: 0-informal, 1-formal

For examples, you can run:
```shell
# for positive sentences
bash conditional_generation.sh 1 1
# for past sentences
bash conditional_generation.sh 4 0
# for positive & future sentences
bash conditional_generation.sh '1,4' '1,2'
# for positive & future & informal
bash conditional_generation.sh '1,4,33' '1,2,0'
# for positive & future & informal and negative & future & informal
bash conditional_generation.sh '1,4,33' '1,2,0;0,2,0'
```
The generated files can be found in *../ckpts/model/sample/* (default: *../ckpts/large_yelp/sample/sampling\*.txt*)

## Train VAE
Modify the path of data file in *code/train_vae.sh*
```shell
dataset=your_dataset_name
# e.g., dataset=yelp
TRAIN_FILE=path_to_train_data_file 
# e.g., TRAIN_FILE=../data/datasets/yelp_data/train.shuf.merge
TEST_FILE=path_to_test_data_file
# e.g., TEST_FILE=../data/datasets/yelp_data/test.merge
```
The structure of the data file: one line one sentence. See *../data/datasets/yelp_data/test.merge* for example.

Then run the script to train a VAE
```shell
cd code
bash train_vae.sh
```
The checkpoints will be saved in *../ckpts/LM/$dataset/$name* by default. You also can find the tensorboard logs in *code/runs/$dataset*
## Train GAN and Classifiers
After training VAE, you can train the GAN and classifiers to do some operations.
### Train GAN
You need to specify some key arguments: 
```shell
train_cls_gan='gan'

ckpt_path=path_to_vae_ckpts  # e.g., ckpt_path=../ckpts/base_yelp

TRAIN_FILE=path_to_test_gan_data_file 
# e.g., TRAIN_FILE=../data/datasets/yelp_data/train_gan.txt

TRAIN_FILE=path_to_test_gan_data_file 
# e.g., TEST_FILE=../data/datasets/yelp_data/test.txt
```
The GAN training and test data file should have the line format (exclude bracket []): [0]\t[text], where the [0] is not used and meaningless in the training and it can be any other integer. See the example in *../data/datasets/yelp_data/train_gan.txt*

Then run the below command to train GAN:
```shell
cd code
bash train_classifier_latent.sh
``` 
### Train Classifiers
You need to specify some key arguments: 
```shell
train_cls_gan='cls'

ckpt_path=path_to_vae_ckpts  # e.g., ckpt_path=../ckpts/base_yelp

TRAIN_FILE=path_to_test_cls_data_file 
# e.g., TRAIN_FILE=../data/datasets/yelp_data/train_sentiment.txt

TRAIN_FILE=path_to_test_cls_data_file 
# e.g., TEST_FILE=../data/datasets/yelp_data/test_sentiment.txt

cls_step=identifier_of_classifier
# identifier of this classifier, the classifier will be stored in path_to_vae_ckpts/checkpoint-cls-1 if cls_step=1

n_classes=number_of_classes
# number of classes,  e.g., if it contains 2 classes, n_classes=2
```
The Classifiers training and test data file should have the line format (exclude bracket []):[class_label]\t[text], where [class_label] should be the class label of the text, it should be a integer. If you have 2 classes, the [class_label] should be 0 or 1. If you have 3 classes, it should be 0, 1, or 2. See *../data/datasets/yelp_data/train_sentiment.txt* for example.

Then run the below command to train Classifiers:
```shell
cd code
bash train_classifier_latent.sh
``` 

## Outputs
To facilitate comparison, we provide the output files of text editing with single attribute (text style transfer) in [*./outputs*](/outputs) folder.


## Cite
```
@misc{liu2022composable,
      title={Composable Text Controls in Latent Space with ODEs}, 
      author={Guangyi Liu and Zeyu Feng and Yuan Gao and Zichao Yang and Xiaodan Liang and Junwei Bao and Xiaodong He and Shuguang Cui and Zhen Li and Zhiting Hu},
      year={2022},
      eprint={2208.00638},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

