root_path="/data/yaser/data/research/Diverse_Weather_Dataset"
echo "输出valid"

# shellcheck disable=SC2128
# shellcheck disable=SC2034
#for dataset_name in "Daytime-Foggy" "Daytime-sunny" "Dusk-rainy" "Night-Sunny" "Night-rainy"; do
for dataset_name in "Daytime-sunny"; do
  echo "输出$dataset_name"
  python voc2coco.py --xml_dir $root_path/"$dataset_name"/valid/voc --json_file $root_path/"$dataset_name"/valid/coco_valid.json --img_dir $root_path/"$dataset_name"/valid/images
done
#echo "输出train"
#python voc2coco.py --xml_dir $root_path/Daytime-sunny/train/voc --json_file $root_path/Daytime-sunny/train/coco_train.json --img_dir $root_path/Daytime-sunny/train/images
