mkdir -p classifiers
wget -c -t 10 "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/ERsrW_eXaJBLtkFuOdLJYJQBLCfqrp8T1Afh9srWRKTOAw?e=LWtKR4&download=1" -O classifiers/classifiers.tar.gz
cd classifiers
tar -zxvf classifiers.tar.gz
cd ..

echo "Classifiers are in ./classifiers"

