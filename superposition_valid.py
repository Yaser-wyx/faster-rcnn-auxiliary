#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.layers import ShapeSpec
from detectron2.modeling import build_model, GeneralizedRCNN
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from src.random_auxiliary_init import RandomAuxiliaryInit, set_random_auxiliary_init
from src.resnet_superposition import build_resnet_superposition, ResNetSuperposition

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

logger = logging.getLogger("detectron2")

distributed = False


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


def get_evaluator(cfg, dataset_name, output_folder=None):
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
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, iteration=0):
    model.train(False)
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
            # output the result to csv
            csv_path = f"{cfg.OUTPUT_DIR}/result.csv"
            if not os.path.exists(csv_path):
                with open(csv_path, "a+") as f:
                    f.write("dataset-name,AP,AP50,AP75,APs,APm,APl\n")
            APs = results_i["bbox"]
            with open(csv_path, "a+") as f:
                ap_str = [dataset_name] + [str(round(APs[key], 4)) for key in
                                             ["AP", "AP50", "AP75", "APs", "APm", "APl"]]
                f.write(",".join(ap_str) + "\n")

    if len(results) == 1:
        results = list(results.values())[0]
    model.train()
    return results


def generate_progressive_strategy(random_layers, feature_lambda, superposition_start, epochs):
    random_layer_list = [i for i in range(1, random_layers + 1, 1)]
    feature_lambda_list = [round(i, 2) for i in np.arange(0.01, feature_lambda + 0.001, 0.01)]
    assert epochs > superposition_start + max(len(random_layer_list), len(feature_lambda_list))

    progressive_strategy_map = {
        "random_layers": {
            superposition_start + i: v for i, v in enumerate(random_layer_list)
        },
        "feature_lambda": {
            superposition_start + i: v for i, v in enumerate(feature_lambda_list)
        },
    }
    logger.info(f"progressive_strategy: {progressive_strategy_map}")
    return progressive_strategy_map


def do_train(cfg, model, auxiliary_model=None, resume=False):
    model.train()

    # prepare for resnet superposition
    if distributed:
        resnet_superposition: ResNetSuperposition = model.module.backbone.bottom_up
    else:
        resnet_superposition: ResNetSuperposition = model.backbone.bottom_up

    use_resnet_superposition = isinstance(resnet_superposition, ResNetSuperposition)
    p = cfg.AUXILIARY.P  # 应用叠加的概率
    superposition_start = cfg.AUXILIARY.SUPERPOSITION_START
    progressive = cfg.AUXILIARY.PROGRESSIVE
    feature_lambda = cfg.AUXILIARY.FEATURE_LAMBDA
    random_layers = cfg.AUXILIARY.RANDOM_LAYERS
    max_iter = cfg.SOLVER.MAX_ITER
    auxiliary_type = cfg.AUXILIARY.AUXILIARY_TYPE
    superposition_way = cfg.AUXILIARY.SUPERPOSITION_WAY

    if auxiliary_type == "Random":
        logging.info("设置随机初始化参数")
        random_auxiliary_init: RandomAuxiliaryInit = set_random_auxiliary_init(
            cfg.AUXILIARY.INITIAL_DISTRIBUTION_NUM,
            cfg.AUXILIARY.RANDOM_AUXILIARY_INITIAL_WAY,
            cfg.AUXILIARY.INIT_DISTRIBUTION_FIXED)
        random_init_batches = cfg.AUXILIARY.INIT_BEFORE_BATCHES

    if superposition_way is not None and progressive:
        progressive_strategy_map = generate_progressive_strategy(random_layers, feature_lambda, superposition_start,
                                                                 max_iter)
    if use_resnet_superposition:
        resnet_superposition.set_superposition_parameter(False)  # 初始化，关闭特征叠加

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            print(f"iteration: {iteration}")
            if use_resnet_superposition:
                if (
                        superposition_way is not None
                        and iteration >= superposition_start
                        and random.random() <= p
                ):
                    # 使用特征叠加的方式，且满足使用条件
                    # 给resnet_superposition设置参数
                    if progressive:
                        # 如果使用递进式，则重新根据生成的map赋值
                        random_layers = progressive_strategy_map["random_layers"].get(iteration, random_layers)
                        feature_lambda = progressive_strategy_map["feature_lambda"].get(iteration, feature_lambda)

                    auxiliary_output = None
                    if auxiliary_type is not None:
                        print("set for auxiliary model")
                        # 设置辅助网络的参数
                        if auxiliary_type == "Random":
                            # random init the auxiliary network
                            if iteration % random_init_batches == 0:
                                print("重新初始化随机辅助网络")
                                assert auxiliary_model is not None
                                random_auxiliary_init.apply_random_init(auxiliary_model)
                        elif auxiliary_type == "Last":
                            pass
                        if distributed:
                            images = model.module.preprocess_image(data)
                        else:
                            images = model.preprocess_image(data)
                        auxiliary_output = auxiliary_model(images.tensor)
                    print(f"resnet_superposition.set_superposition_parameter")
                    resnet_superposition.set_superposition_parameter(True, feature_lambda=feature_lambda,
                                                                     random_layers=random_layers,
                                                                     auxiliary_output=auxiliary_output)
                else:
                    resnet_superposition.set_superposition_parameter(False)

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type=2)

            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                do_test(cfg, model, iteration)
                # Compared to "train.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


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


def main(args):
    register_custom_dataset()
    global distributed
    cfg = setup(args)
    with_auxiliary = cfg.AUXILIARY.AUXILIARY_TYPE is not None
    resnet_auxiliary_model = None
    if with_auxiliary:
        resnet_auxiliary_model = build_resnet_superposition(cfg, ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN)),
                                                            is_auxiliary=True)
        # logger.info("build resnet auxiliary model:\n{}".format(resnet_auxiliary_model))
        resnet_auxiliary_model.to(torch.device(cfg.MODEL.DEVICE))
    else:
        logger.info("without auxiliary network!")

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
        if with_auxiliary:
            resnet_auxiliary_model = DistributedDataParallel(
                resnet_auxiliary_model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

    do_train(cfg, model, auxiliary_model=resnet_auxiliary_model, resume=args.resume)
    return do_test(cfg, model)


def auxiliary_argument_parser(parser):
    parser.add_argument('--auxiliary-type', type=str, choices=['Random', 'Last', 'None'], default='None',
                        help='use auxiliary model to help training')
    parser.add_argument('--progressive', action='store_true',
                        help='use linear progressive strategy to train model with auxiliary net')
    parser.add_argument('--random-layers', type=int, default=0)
    # 代表叠加图像所占的比例
    parser.add_argument('--feature-lambda', type=float, default=0.)

    parser.add_argument('--superposition-way', type=str, choices=['MixStyle', 'Direct', 'None'], default='None',
                        help='the way of superposition for the auxiliary network')
    parser.add_argument('--initial-distribution-num', type=int, default=0,
                        help='the number of init distribution for random auxiliary network to init')
    parser.add_argument('--init-before-batches', type=int, default=0,
                        help='how many batches before random auxiliary network to re-init')
    parser.add_argument('--p', type=float, default=0.,
                        help='the probability of use superposition')
    parser.add_argument('--random-auxiliary-initial-way', type=str, choices=['whole-net', 'per-layer'],
                        default='per-layer',
                        help='the initial way of random-auxiliary')

    parser.add_argument('--init-distribution-fixed', action='store_true', help='fixed the initial distribution')

    parser.add_argument('--apply-net-layers', type=int, default=0)
    parser.add_argument('--superposition-start', type=int, default=0)
    return parser


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
