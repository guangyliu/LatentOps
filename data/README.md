# Dataset
We provide Yelp review dataset and Amazon review dataset in *datasets* folder.

## For Latent Model (VAE) Training 
- *train.merge*: All training data without sentiment labels 
- *train.shuf.merge*: All training data without sentiment labels (Shuffled order) 
- *test.merge*: All test data without sentiment labels

## For Operators (Classifiers)
*[attribute]* can be *sentiment* and *tense*. *[keyword]* can be any one keyword from all 613 keywords (please refer keyword list in our paper).
- *train_[attribute].txt*: Randomly selected training samples with sentiment labels (200 samples per class).
- *test_[attribute].txt*: Randomly selected test samples with sentiment labels.
- *target_word/[keyword]_train.txt*: Keyword related training samples (200 samples per class).
- *target_word/[keyword]_test.txt*: Keyword related test samples (20 samples per class).


## For Better Initialization (GAN)
- *train_gan.txt*: Randomly selected training samples without labels for training GAN.