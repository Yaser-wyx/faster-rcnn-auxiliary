out_dir=runs/train/RandomAuxiliaryDistributionFixedOne
apply_net_layers=2
random_layers=1

for distribution_name in "xavier_normal_" "orthogonal_" "trunc_normal_" "normal_"; do
  python superposition_trainer.py --progressive --auxiliary-type Random \
    --superposition-way MixStyle --p 0.7 --superposition-start 500 \
    --feature-lambda 0.9 --random-layers $random_layers --apply-net-layers $apply_net_layers \
    --init-before-batches 50 --num-gpus 2 --initial-distribution-num 1 \
    --config-file configs/Generalization/faster_rcnn_R_101_FPN_3x_Diverse_Weather_paper.yaml \
    --random-auxiliary-initial-way per-layer \
    --fixed-distribution-list $distribution_name \
    --GPU 0,1 \
    OUTPUT_DIR $out_dir/d@$distribution_name
done
