# data/

Run `python data/prepare_data.py` (or `bash download_datasets.sh` from the repo root) to
build `data/datasets/yelp_data/` and `data/datasets/amazon_data/` from the public
Li et al. (2018) corpora. Both are gitignored.

| File | Format | Used by |
|---|---|---|
| `train.merge`, `train.shuf.merge` | one sentence per line | VAE training (`train_vae.sh`) |
| `test.merge` | one sentence per line | VAE evaluation |
| `train_sentiment.txt`, `test_sentiment.txt` | `<label>\t<text>`, 0 = negative, 1 = positive; 200 training samples per class | latent classifier (`train_classifier_latent.sh`, `train_cls_gan=cls`) |
| `train_gan.txt` | `0\t<text>` (label unused) | latent GAN (`train_classifier_latent.sh`, `train_cls_gan=gan`) |
| `test_ref.txt` | `<label>\t<text>`, 500 negative then 500 positive | style-transfer input (`lace_transfer_yelpnew.sh`) |
| `reference.0`, `reference.1` | one human-written reference per line, aligned with `test_ref.txt` | BLEU evaluation |
| `target_word/<kw>_{train,test}.txt` | `<label>\t<text>`, 1 = contains keyword | keyword operators (`--keywords`) |

Tense labels (past / present / future) were produced with a POS-tagger heuristic and the
formality operator was trained on GYAFC, which requires a data agreement with Yahoo; neither
is rebuilt by the script. Any `<label>\t<text>` file with integer labels works with the
classifier training script, so you can supply your own.

Source corpora: Yelp and Amazon reviews as preprocessed and released by
Li, Jia, He and Liang, *Delete, Retrieve, Generate* (NAACL 2018), CC BY-SA 4.0.
