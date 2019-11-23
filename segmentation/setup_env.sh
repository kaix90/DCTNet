# Configure environment
sudo apt install -y unzip
mkdir -p /mnt/dataset/coco
cd /mnt/dataset/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip
rm train2017.zip
rm val2017.zip
rm test2017.zip

cd /mnt/kai/work/code
git clone https://github.com/calmevtime/dctDet
mkdir -p /mnt/kai/work/code/dctDet/data/
cd /mnt/kai/work/code/dctDet/data/
ln -s /mnt/dataset/coco/ /mnt/kai/work/code/dctDet/data/
