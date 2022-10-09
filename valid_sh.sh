# shellcheck disable=SC2034
#root_path="/data/yaser/data/research/Diverse_Weather_Dataset"
model_path="runs/train/RandomAuxiliaryDistributionFixedOne"
out_path="runs/valid/RandomAuxiliaryDistributionFixedOne"
# shellcheck disable=SC2043
for model_name in "d@kaiming_normal_" "d@xavier_normal_"; do
  echo "model_name:$model_name"
  python superposition_valid.py --eval-only \
    --num-gpus 2 \
    --config-file configs/Generalization/faster_rcnn_R_101_FPN_3x_Diverse_Weather_test.yaml \
    OUTPUT_DIR $out_path/$model_name \
    MODEL.WEIGHTS $model_path/$model_name/model_final.pth
done
