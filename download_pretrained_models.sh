mkdir -p ckpts
echo "Download base_yelp.tar.gz"

wget "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/EeLMgN-7D-VAqpOZHWTS2-YBW6ArjF8_qdPvyAM9-SCfmg?e=MuqbaF&download=1" -O ckpts/base_yelp.tar.gz


echo "Download large_yelp.tar.gz"

wget "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/EQ58ZvN4dCxChewzsu9Nkc8BqiwVKRmlTtNnEo0G8kfH6A?e=ApoZPv&download=1" -O ckpts/large_yelp.tar.gz


echo "Download large_amazon.tar.gz"

wget "https://cuhko365-my.sharepoint.com/:u:/g/personal/218019026_link_cuhk_edu_cn/EfwA6NXJpOBNi8S5CxxzbLMBmflZawkyPU-8ThxkWtBNFA?e=NMFUfm&download=1" -O ckpts/large_amazon.tar.gz

cd ckpts
tar -zxvf base_yelp.tar.gz
tar -zxvf large_yelp.tar.gz
tar -zxvf large_amazon.tar.gz
cd ..

