import os
import json
import pandas as pd


class HyperSolver(object):
    CLSNAMES = [
        'colon',
    ]

    def __init__(self, root='data/mvtec'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}'
            for phase in ['test']:
                cls_info = []
                species = {
                    name for name in os.listdir(cls_dir)
                    if name != 'masks' and os.path.isdir(os.path.join(cls_dir, name))
                }
                print("species", species)
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_names = os.listdir(f'{cls_dir}/{specie}')
                    mask_names = os.listdir(f'{cls_dir}/masks/') if is_abnormal else None
                    img_names.sort()
                    mask_names.sort() if mask_names is not None else None
                    assert len(img_names) == len(mask_names) if mask_names is not None else True
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{specie}/{img_name}',
                            mask_path=f'masks/{mask_names[idx]}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples = anomaly_samples + 1
                            else:
                                normal_samples = normal_samples + 1
                    info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)



if __name__ == '__main__':
    from _root import parse_dataset_root
    runner = HyperSolver(root=parse_dataset_root('EndoTect'))
    runner.run()
