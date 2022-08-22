mkdir -p data
wget -c -t 10 "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/ETzJ0Fae4-lHi3vN8G8HYbQBvZr7wh7iQvqMCd2YloAb_g?e=i4NE7O&download=1" -O data/datasets.tar.gz
cd data
tar -zxvf datasets.tar.gz
cd ..

echo "Datasets are in ./data/datasets"

