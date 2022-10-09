#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from src.resnet_superposition import build_resnet_superposition_fpn_backbone

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    print(f"evaluator_type: {evaluator_type}")
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    auxiliary_setup(cfg, args)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def auxiliary_setup(cfg, args):
    cfg.AUXILIARY = CfgNode()
    arg_name_list = ["feature_lambda", "superposition_start", "auxiliary_type", "random_layers", "progressive",
                     "apply_net_layers", "superposition_way", "initial_distribution_num", "init_before_batches",
                     "random_auxiliary_initial_way", "init_distribution_fixed", "p"]
    for arg_name in arg_name_list:
        cfg.AUXILIARY[arg_name.upper()] = getattr(args, arg_name)
    # 将None的文本转为None值
    for k, v in cfg.AUXILIARY.items():
        if v == "None":
            cfg.AUXILIARY[k] = None


def register_custom_dataset():
    # register dataset for Generalization
    train_dataset_map = {  # for train
        "SIM10K": "/data/yaser/data/research/SIM_10K_Dataset",  # only car
        "Daytime-sunny": "/data/yaser/data/research/Diverse_Weather_Dataset/Daytime-sunny"
    }
    valid_dataset_map = {  # for valid
        "SIM10K": "/data/yaser/data/research/SIM_10K_Dataset",  # only car
        "Cityscape-raw": "/data/yaser/data/research/CityscapeDataset_car/rawDataset",  # only car
        "Cityscape-rain": "/data/yaser/data/research/CityscapeDataset_car/rainDataset",  # only car
        "Cityscape-foggy": "/data/yaser/data/research/CityscapeDataset_car/foggyDataset",  # only car
        "Daytime-sunny": "/data/yaser/data/research/Diverse_Weather_Dataset/Daytime-sunny",
        "Daytime-Foggy": "/data/yaser/data/research/Diverse_Weather_Dataset/Daytime-Foggy",
        "Dusk-rainy": "/data/yaser/data/research/Diverse_Weather_Dataset/Dusk-rainy",
        "Night-Sunny": "/data/yaser/data/research/Diverse_Weather_Dataset/Night-Sunny",
        "Night-rainy": "/data/yaser/data/research/Diverse_Weather_Dataset/Night-rainy",
        "target_domain_mix": "/data/yaser/data/research/Diverse_Weather_Dataset/target_domain_mix",
    }

    test_dataset_map = {
        "Daytime-sunny": "/data/yaser/data/research/Diverse_Weather_Dataset/Daytime-sunny",
        "Daytime-Foggy": "/data/yaser/data/research/Diverse_Weather_Dataset/Daytime-Foggy",
        "Dusk-rainy": "/data/yaser/data/research/Diverse_Weather_Dataset/Dusk-rainy",
        "Night-Sunny": "/data/yaser/data/research/Diverse_Weather_Dataset/Night-Sunny",
        "Night-rainy": "/data/yaser/data/research/Diverse_Weather_Dataset/Night-rainy",
    }
    for dataset_name in train_dataset_map:
        dataset_name_with_stage = f"{dataset_name}_train"
        dataset_root = train_dataset_map[dataset_name]
        register_coco_instances(dataset_name_with_stage, {},
                                f"{dataset_root}/train/coco_train.json",
                                f"{dataset_root}/train/images", )
    for dataset_name in valid_dataset_map:
        dataset_name_with_stage = f"{dataset_name}_valid"
        dataset_root = valid_dataset_map[dataset_name]
        register_coco_instances(dataset_name_with_stage, {},
                                f"{dataset_root}/valid/coco_valid.json",
                                f"{dataset_root}/valid/images", )

    for dataset_name in test_dataset_map:
        dataset_name_with_stage = f"{dataset_name}_test"
        dataset_root = test_dataset_map[dataset_name]
        register_coco_instances(dataset_name_with_stage, {},
                                f"{dataset_root}/test/coco_valid.json",
                                f"{dataset_root}/test/images", )


def auxiliary_argument_parser(parser):
    parser.add_argument('--auxiliary-type', type=str, choices=['Random', 'Last', 'None'], default='None',
                        help='use auxiliary model to help training')
    parser.add_argument('--progressive', action='store_true',
                        help='use linear progressive strategy to train model with auxiliary net')
    parser.add_argument('--random-layers', type=int, default=0)
    # 如果使用mixstyle叠加方式，则该值代表保留原始图像风格信息的比例，否则就是代表叠加图像所占的比例
    parser.add_argument('--feature-lambda', type=float, default=0.1)

    parser.add_argument('--superposition-way', type=str, choices=['MixStyle', 'Direct', 'None'], default='None',
                        help='the way of superposition for the auxiliary network')
    parser.add_argument('--initial-distribution-num', type=int, default=0,
                        help='the number of init distribution for random auxiliary network to init')
    parser.add_argument('--init-before-batches', type=int, default=5,
                        help='how many batches before random auxiliary network to re-init')
    parser.add_argument('--p', type=float, default=0.,
                        help='the probability of use superposition')
    parser.add_argument('--random-auxiliary-initial-way', type=str, choices=['whole-net', 'per-layer'],
                        default='per-layer',
                        help='the initial way of random-auxiliary')

    parser.add_argument('--init-distribution-fixed', action='store_true', help='fixed the initial distribution')

    parser.add_argument('--apply-net-layers', type=int, default=0)
    parser.add_argument('--superposition-start', type=int, default=5)
    return parser


def main(args):
    register_custom_dataset()
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = auxiliary_argument_parser(default_argument_parser()).parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
