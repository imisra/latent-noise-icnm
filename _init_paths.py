# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths."""
import os.path as osp
import os
import sys
import platform

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        print 'added {}'.format(path)

this_dir = osp.dirname(__file__)
#add utils
add_path(osp.join(this_dir, 'utils'))

#add coco utils
add_path(osp.join(this_dir, 'coco/PythonAPI/'))

# Add caffe to PYTHONPATH
caffe_path = './caffe-icnm'
add_path(caffe_path)
add_path(osp.join(caffe_path, 'python'))
add_path(osp.join(caffe_path, 'src/caffe/')) #for python layers

# Add this directory to PYTHONPATH
root_path = osp.join(this_dir, '.')
add_path(root_path)
