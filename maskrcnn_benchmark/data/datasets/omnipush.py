# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np
import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.center import Center


min_keypoints_per_image = 1


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class OmnipushDataset(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.img_names = glob.glob(os.path.join(root, '*.jpg'))
        self.annos = np.load(ann_file).item()
        self.transforms = transforms

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = Image.open(self.img_names[idx]).convert('RGB')
        anno = self.annos[self.img_names[idx].split('/')[-1]]

        boxes = [anno['bbox']]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xyxy")

        classes = [anno['class']]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        center = [anno['center']]
        center = Center(center, img.size)
        target.add_field("keypoints", center)

        # ensure bbox is legit
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        return {"height": 640, "width": 640}
