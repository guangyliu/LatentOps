#!/usr/bin/env python3
"""Rebuild the LatentOps data files from public sources.

The original `datasets.tar.gz` was hosted on a university SharePoint that is no longer
reachable. Everything it contained for Yelp and Amazon derives from the sentiment
style-transfer corpora released by Li et al. (2018), "Delete, Retrieve, Generate"
(CC BY-SA 4.0), which this script downloads from GitHub and re-assembles into the
layout the training and evaluation scripts expect:

    data/datasets/<name>_data/
        train.merge            all training sentences, no labels        (VAE training)
        train.shuf.merge       the same, shuffled with a fixed seed      (VAE training)
        test.merge             all test sentences, no labels             (VAE eval)
        train_sentiment.txt    "<label>\t<text>", 200 per class          (latent classifier)
        test_sentiment.txt     "<label>\t<text>", the full test set      (latent classifier)
        train_gan.txt          "0\t<text>", 5,000 unlabelled sentences   (GAN init)
        test_ref.txt           "<label>\t<text>", 500 neg + 500 pos      (style-transfer input)
        reference.0/.1         human references from Li et al.           (BLEU vs. reference)
        target_word/<kw>_{train,test}.txt   keyword operators, see --keywords

Labels: 0 = negative, 1 = positive. Tense and formality labels used in the paper were
produced with a POS-tagger heuristic and the GYAFC corpus respectively; they are not
rebuilt here (GYAFC requires a data agreement with Yahoo).

Usage:
    python data/prepare_data.py                    # yelp + amazon
    python data/prepare_data.py --datasets yelp
    python data/prepare_data.py --keywords food service price
"""
import argparse, os, random, urllib.request

RAW = 'https://raw.githubusercontent.com/lijuncen/Sentiment-and-Style-Transfer/master/data'
SPLITS = ['sentiment.train.0', 'sentiment.train.1', 'sentiment.dev.0', 'sentiment.dev.1',
          'sentiment.test.0', 'sentiment.test.1', 'reference.0', 'reference.1']


def fetch(name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for s in SPLITS:
        dst = os.path.join(out_dir, s)
        if os.path.exists(dst):
            continue
        print('  download', f'{name}/{s}')
        urllib.request.urlretrieve(f'{RAW}/{name}/{s}', dst)


def lines(p):
    with open(p, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def write(p, rows):
    with open(p, 'w', encoding='utf-8') as f:
        f.write('\n'.join(rows) + '\n')


def build(name, root, seed, per_class, gan_n, keywords):
    raw = os.path.join(root, 'raw', name)
    out = os.path.join(root, 'datasets', f'{name}_data')
    fetch(name, raw)
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed)

    tr = {c: lines(os.path.join(raw, f'sentiment.train.{c}')) for c in (0, 1)}
    te = {c: lines(os.path.join(raw, f'sentiment.test.{c}')) for c in (0, 1)}

    train_all = tr[0] + tr[1]
    write(os.path.join(out, 'train.merge'), train_all)
    shuf = train_all[:]; rng.shuffle(shuf)
    write(os.path.join(out, 'train.shuf.merge'), shuf)
    write(os.path.join(out, 'test.merge'), te[0] + te[1])

    cls_train = [f'{c}\t{t}' for c in (0, 1) for t in rng.sample(tr[c], per_class)]
    rng.shuffle(cls_train)
    write(os.path.join(out, 'train_sentiment.txt'), cls_train)
    write(os.path.join(out, 'test_sentiment.txt'), [f'{c}\t{t}' for c in (0, 1) for t in te[c]])

    write(os.path.join(out, 'train_gan.txt'), [f'0\t{t}' for t in rng.sample(train_all, gan_n)])

    # Style-transfer evaluation input: the 500 + 500 Li et al. test sentences, in the same
    # order as outputs/style_transfer/<name>/latentops.txt and reference.0/1.
    write(os.path.join(out, 'test_ref.txt'), [f'{c}\t{t}' for c in (0, 1) for t in te[c]])
    for c in (0, 1):
        ref = lines(os.path.join(raw, f'reference.{c}'))
        write(os.path.join(out, f'reference.{c}'), ref)

    if keywords:
        kw_dir = os.path.join(out, 'target_word'); os.makedirs(kw_dir, exist_ok=True)
        for kw in keywords:
            pos = [t for t in train_all if f' {kw} ' in f' {t} ']
            neg = [t for t in train_all if f' {kw} ' not in f' {t} ']
            if len(pos) < per_class + 20:
                print(f'  [skip] keyword "{kw}": only {len(pos)} sentences contain it')
                continue
            p, n = rng.sample(pos, per_class + 20), rng.sample(neg, per_class + 20)
            train = [f'1\t{t}' for t in p[:per_class]] + [f'0\t{t}' for t in n[:per_class]]
            test = [f'1\t{t}' for t in p[per_class:]] + [f'0\t{t}' for t in n[per_class:]]
            rng.shuffle(train)
            write(os.path.join(kw_dir, f'{kw}_train.txt'), train)
            write(os.path.join(kw_dir, f'{kw}_test.txt'), test)
    print(f'  {name}: train {len(train_all):,}  test {len(te[0]) + len(te[1]):,}  -> {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=['yelp', 'amazon'], choices=['yelp', 'amazon'])
    ap.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--per_class', type=int, default=200, help='classifier training samples per class')
    ap.add_argument('--gan_n', type=int, default=5000)
    ap.add_argument('--keywords', nargs='*', default=[], help='build keyword operator files for these words')
    a = ap.parse_args()
    for d in a.datasets:
        print('==', d)
        build(d, a.root, a.seed, a.per_class, a.gan_n, a.keywords)
