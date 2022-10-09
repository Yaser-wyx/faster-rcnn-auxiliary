import os


def run(random_layers=1, apply_net_layers=1, num_gpus=2, project_name="exp"):
    cmd = f"python superposition_trainer.py --progressive --auxiliary-type Random " \
          f"--superposition-way MixStyle --p 0.7 --superposition-start 500 " \
          f"--feature-lambda 0.9 --random-layers {random_layers} --apply-net-layers {apply_net_layers}  " \
          f"--init-before-batches 50 --num-gpus {num_gpus} --initial-distribution-num 7 " \
          f"--config-file configs/Generalization/faster_rcnn_R_101_FPN_3x_Diverse_Weather_paper.yaml " \
          f"--random-auxiliary-initial-way per-layer " \
          f"OUTPUT_DIR runs/train/RandomAuxiliary/{project_name}"
    os.system(cmd)


