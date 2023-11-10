"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import json
import random as r
import os


class MyDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        labels = ['handbag', 'suitcase', 'stroller', 'backpack', 'golf_club', 'duffel_bag',]
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'detections': tfds.features.Sequence({
                    'label': tfds.features.ClassLabel(
                        names=labels,
                    ),
                    'bbox': tfds.features.BBoxFeature(),
                })
            }),
            supervised_keys=('image', 'detections'),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Download source data
        parent_path = ''
        dataset_path = os.path.join(parent_path, '/data/dataset')
        return {
            'train': self._generate_examples(self, dataset_path = dataset_path,)
        }
    
    def _generate_examples(self, dataset_path):
        annotations_path=dataset_path + '/!annotations.json'
        # Read the input data out of the source files
        
        with open(annotations_path, encoding='utf-8') as f:
            j = json.load(f)
            for key,value in j['_via_img_metadata'].items():
                if not value['regions']:
                    continue
                image = dataset_path + '/' + value['filename']
                (ht, wd, _) = self.get_img_size(self, image)
                detections = []
                for r in value['regions']:
                    label = r['region_attributes']['class']

                    b = r['shape_attributes']
                    y1 = max(0, b['y'])
                    x1 = max(0, b['x'])
                    y2 = min(ht, b['y'] + b['height'])
                    x2 = min(wd, b['x'] + b['width'])
                    bbox = tfds.features.BBox(
                        ymin=y1 / ht,
                        xmin=x1 / wd,
                        ymax=y2 / ht,
                        xmax=x2 / wd,
                    )

                    detections.append({'label': label, 'bbox': bbox})
                         
                yield key, {
                    'image': image,
                    'detections': detections,
                }
    
    def get_img_size(self, filename):
        raw = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(raw, channels=3)
        return image.shape
    
    def normalize_xywh(im_x, im_y, x:[], y:[]):
        x_norm = []
        y_norm = []
        for i in x:
            x_norm.append(i / im_x)
        for j in y:
            y_norm.append(j / im_y)
        
        return x_norm,y_norm
        
        
        

"""
test_builder = MyDatasetBuilder
data = test_builder._split_generators(self=test_builder, dl_manager=tfds.download.DownloadManager)
num_det = 0
num_imgwdet = 0
for k,v in data['train']:
    num_imgwdet += 1
    num_det += len(v['detections'])
print('Images with detections: ', num_imgwdet)
print('Total detections: ', num_det)
"""