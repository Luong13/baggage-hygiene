"""my_dataset dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import keras_cv
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
        labels = ['handbag', 'suitcase', 'stroller', 'backpack', 'golf_club', 'duffel_bag']
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(),
                'bounding_boxes': tfds.features.Sequence({
                    "boxes": tfds.features.BBoxFeature(),
                    "classes": tfds.features.ClassLabel(names=labels),
                }),
            }),
            supervised_keys=('image', 'bounding_boxes'),
        )

    def _split_generators(self, dl_manager=None):
        # Download source data
        parent_path = ''
        dataset_path = os.path.join(parent_path, '/data/dataset')
        return {'train':self._generate_examples(dataset_path = dataset_path,)}
    
    def _generate_examples(self, dataset_path, bounding_box_format="xywh"):
        # Read the input data out of the source files
        with open(dataset_path + '/!annotations.json', encoding='utf-8') as f:
            j = json.load(f)
            for key,value in j['_via_img_metadata'].items():
                image = dataset_path + '/' + value['filename']
                if not value['regions']:
                    continue
                if not os.path.isfile(image):
                    continue
                
                (ht, wd, _) = self.get_img_size(image)
                detections = []
                for r in value['regions']:
                    label = r['region_attributes']['class']

                    b = r['shape_attributes']
                    y1 = max(0, b['y'])
                    x1 = max(0, b['x'])
                    y2 = min(ht, y1 + b['height'])
                    x2 = min(wd, x1 + b['width'])          
                    boxes = tfds.features.BBox(        
                        y1 / ht,
                        x1 / wd,
                        y2 / ht,
                        x2 / wd,
                    )

                    detections.append({'boxes':boxes,'classes':label})
                         
                yield key, {
                    'image': image,
                    'bounding_boxes': detections,
                }
    
    def get_img_size(self, filename):
        raw = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(raw, channels=3)
        return image.shape