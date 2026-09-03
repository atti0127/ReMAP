import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    key = dataset_name.lower()
    aliases = {
        'mvtec': ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                  'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood'],
        'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'],
        'mpdd': ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'],
        'btad': ['01', '02', '03'],
        'dagm': ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10'],
        'dagm_kaggleupload': ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6', 'Class7', 'Class8', 'Class9', 'Class10'],
        'sdd': ['electrical commutators'],
        'dtd': ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069'],
        'dtd-synthetic': ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069'],
        'headct': ['brain'],
        'brainmri': ['brain'],
        'br35h': ['brain'],
        'brain': ['brain'],
        'covid-19': ['chest'],
        'covid19': ['chest'],
        'chest': ['chest'],
        'isic': ['skin'],
        'isbi': ['skin'],
        'cvc-colondb': ['colon'],
        'cvc-clinicdb': ['colon'],
        'kvasir': ['colon'],
        'endo': ['colon'],
        'colon': ['colon'],
        'tn3k': ['thyroid'],
        'thyroid': ['thyroid'],
    }
    if key not in aliases:
        raise ValueError('unsupported dataset identifier: {}'.format(dataset_name))
    obj_list = aliases[key]
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        with open(f'{self.root}/meta.json', 'r') as handle:
            meta_info = json.load(handle)
        name = self.root.split('/')[-1]
        phase = next((key for key in meta_info if key.lower() == mode.lower()), None)
        if phase is None:
            raise KeyError("metadata has no '{}' split: {}".format(mode, sorted(meta_info)))
        meta_info = meta_info[phase]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}
