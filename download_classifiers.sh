mkdir -p classifiers
wget -c -t 10 "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/ERsrW_eXaJBLtkFuOdLJYJQBLCfqrp8T1Afh9srWRKTOAw?e=LWtKR4&download=1" -O classifiers/classifiers.tar.gz
wget -c -t 10 "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/EeyDq8jASiZPjhKoVqG1m9EBgI1Yz72vaCiiakweCYERYA?e=D8hMpQ&download=1" -O classifiers/gpt2_yelp.tar.gz
cd classifiers
tar -zxvf classifiers.tar.gz
tar -zxvf gpt2_yelp.tar.gz
mv gpt2_yelpshort gpt2_yelp
cd ..

echo "Classifiers are in ./classifiers"

