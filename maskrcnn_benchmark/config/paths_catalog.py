# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "/data0/qilei_chen/"

    DATASETS = {
        "coco_rop_ridge_2014_train": (
            "AI_EYE/ROP_DATASET/9LESIONS/train2014",
            "AI_EYE/ROP_DATASET/9LESIONS/annotations/ridge_in_one_instances_train2014.json",
        ),
        "coco_rop_ridge_2014_val": (
            "AI_EYE/ROP_DATASET/9LESIONS/val2014",
            "AI_EYE/ROP_DATASET/9LESIONS/annotations/ridge_in_one_instances_val2014.json",
        ),
        "coco_dr4lesions_2014_train": (
            "AI_EYE/BostonAI4DB7/train2014",
            "AI_EYE/BostonAI4DB7/annotations/instances_train2014.json",
        ),
        "coco_dr4lesions_2014_val": (
            "AI_EYE/BostonAI4DB7/val2014",
            "AI_EYE/BostonAI4DB7/annotations/instances_val2014.json",
        ),
        "coco_test-dev": (
            "MSCOCO2017/images",
            "MSCOCO2017/annotations/image_info_test-dev2017.json",
        ),
        "coco_2017_test": (
            "MSCOCO2017/images",
            "MSCOCO2017/annotations/image_info_test2017.json",
        ),
        "coco_2017_train": (
            "MSCOCO2017/images",
            "MSCOCO2017/annotations/instances_train2017.json",
        ),
        "coco_2017_val": (
            "MSCOCO2017/images",
            "MSCOCO2017/annotations/instances_val2017.json",
        ),
        "coco_2014_train": (
            "coco/train2014",
            "coco/annotations/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "coco/val2014",
            "coco/annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco/val2014",
            "coco/annotations/instances_valminusminival2014.json",
        ),
        "voc_2007_train": (
            "VOC2007/JPEGImages",
            "VOC2007/annotations/voc_2007_train.json",
        ),
        "voc_2007_val": (
            "VOC2007/JPEGImages",
            "VOC2007/annotations/voc_2007_val.json",
        ),
        "voc_2007_test": (
            "VOC2007/JPEGImages",
            "VOC2007/annotations/voc_2007_test.json",
        ),
        "voc_2012_train": (
            "VOC2012/JPEGImages",
            "VOC2012/annotations/voc_2012_train.json",
        ),
        "voc_2012_val": (
            "VOC2012/JPEGImages",
            "VOC2012/annotations/voc_2012_val.json",
        ),
    }

    @staticmethod
    def get(name):
        if "coco" in name or "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
