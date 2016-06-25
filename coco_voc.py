# --------------------------------------------------------
# Written by Saurabh Gupta
# Modified by Ishan Misra
# --------------------------------------------------------

import math, sys, json, os, h5py
import numpy as np
from pycocotools.coco import COCO
import code

class coco_voc():
  def __init__(self, dataset, image_set, image_path=None, captionJsonPath = None):
    self._name = dataset
    self._image_set = image_set
    self._image_path = image_path;
    # Load the annotation file
    if not captionJsonPath:
      captionJsonPath = COCO(os.path.join(self._devkit_path, 'annotations',\
       'captions_trainval2014.json'));
    self._coco_caption_data = COCO(captionJsonPath);

    with open(captionJsonPath,'r') as fh:
      self._captionsData = json.load(fh);

    if dataset == 'coco':
      if image_set in ['train','val','test']:
        self._captionsData['imDir'] = '%s2014'%(image_set);
      elif image_set in ['valid1','valid2', 'valid2-1']:
        self._captionsData['imDir'] = 'val2014'
      else:
        raise ValueError('uknown set for coco: %s' %(image_set))
    elif dataset in ['yfcc']:
        self._captionsData['imDir'] = '';

    # print self._captionsData;
    self._parse_stories();
    self._parse_captions();

    self._image_ext = '.jpg'
    image_index_str, self._image_index = self._load_image_set_index()
    self._image_source_data = self._load_image_set_source();
    self._image_keys = self._load_image_keys();
    self._image_bnames = self._load_image_bnames();
    self._image_ids = self._load_image_ids();


  def get_all_image_bnames_ids(self):
    return self._image_bnames, self._image_ids

  def image_path_from_id(self, im_id):
    """
    Construct an image path from the image's "index" identifier.
    """  
    index = self._image_ids.index(im_id);
    if 'source_data' not in self._captionsData:
      dirpath = os.path.join(self._image_path, self._captionsData['imDir']);
    else:
      dirpath = self._captionsData['source_data'][self._image_source_data[index]];

    image_path = os.path.join(dirpath, self._image_bnames[index]);  
    assert os.path.exists(image_path), \
        'Path does not exist: {}'.format(image_path)
    return image_path

  def image_bname_from_id(self, im_id):
    """
    Construct an image path from the image's "index" identifier.
    """  
    index = self._image_ids.index(im_id);
    return self._image_bnames[index];

  def image_path_from_bname(self, bname):
    """
    Construct an image path from the image's "index" identifier.
    """
    if 'source_data' not in self._captionsData:
      dirpath = os.path.join(self._image_path, self._captionsData['imDir']);
    else:
      index = self._image_bnames.index(bname);
      dirpath = self._captionsData['source_data'][self._image_source_data[index]];
    image_path = os.path.join(dirpath, bname);
    assert os.path.exists(image_path), \
        'Path does not exist: {}'.format(image_path)
    return image_path  

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    image_index = [x['id'] for x in self._captionsData['images']];
    imlist = [int(x) for x in image_index]
    return image_index, imlist

  def has_sources(self):
    return self._image_source_data is not None;

  def _load_image_set_source(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    if not 'source_data' in self._captionsData['images'][0]:
      return None;
    image_source = [x['source_data'] for x in self._captionsData['images']];    
    return image_source;

  def _load_image_keys(self):
    image_keys = [x['file_name'].replace(self._image_ext,'') for x in self._captionsData['images']];
    return image_keys;

  def _load_image_bnames(self):
    image_bnames = [x['file_name'] for x in self._captionsData['images']];
    return image_bnames;

  def _load_image_ids(self):
    image_ids = [x['id'] for x in self._captionsData['images']];
    return image_ids;

  def _get_default_path(self):
    """
    Return the default path where COCO is expected to be installed.
    """
    return os.path.join('/data/coco');

  def _parse_captions(self):
    self._im2caption_index = {};
    self._caption_index = {};
    self._caption2im_index = {};

    for x in self._captionsData['annotations']:
      if x['image_id'] not in self._im2caption_index:
          self._im2caption_index[x['image_id']] = [];
      self._im2caption_index[x['image_id']].append(x['id']);
      self._caption_index[x['id']] = x['caption'];
      self._caption2im_index[x['id']] = x['image_id'];

  def _parse_stories(self):
    if len(self._captionsData['annotations'])<=1 or (not 'story_id' in self._captionsData['annotations'][0]):
      self._has_story = False;
      return;
    self._has_story = True;
    try:
      self._story_ids = list(set([x['story_id'] for x in self._captionsData['annotations']]));
    except:
      self._has_story = False;
      return;

    self._story_ids.sort(); #always deterministic ordering of storyids
    self._story_index = {};
    self._im2story_index = {};
    for x in self._captionsData['annotations']:
      sid = x['story_id'];
      wsid = x['w_story_id'];
      cid = x['id'];
      if not sid in self._story_index:
        self._story_index[sid] = {};        
      self._story_index[sid][wsid] = {'image_id': x['image_id'], 'caption': x['caption'], 'caption_id': cid};
      if x['image_id'] not in self._im2story_index:
        self._im2story_index[x['image_id']] = [];
      self._im2story_index[x['image_id']].append(sid);

      if 'source_data' in x:
        self._story_index[sid][wsid]['source_data'] = x['source_data'];


  def get_caption(self, capt_id):
    return self._caption_index[capt_id];

  def get_caption_image_id(self, capt_id):
    return self._caption2im_index[capt_id];

  def get_image_caption_ids(self, image_id):
    return self._im2caption_index[image_id];

  def get_image_story_ids(self, image_id):
    return self._im2story_index[image_id];

  def get_all_story_ids(self):
    return self._story_ids;    

  def get_story_data(self, story_id):    
    return self._story_index[story_id]

  def get_story_image_ids(self, story_id):
    data =  self._story_index[story_id];
    image_ids = [ data[x]['image_id'] for x in range(len(data.keys())) ];
    return image_ids;

  def get_story_image_bnames(self, story_id):
    image_ids = self.get_story_image_ids(story_id);
    return [self.image_bname_from_id(x) for x in image_ids];  

  def get_story_image_paths(self, story_id):
    image_ids = self.get_story_image_ids(story_id);
    return [self.image_path_from_id(x) for x in image_ids];

  def get_story_caption_ids(self, story_id):
    data =  self._story_index[story_id];
    caption_ids = [ data[x]['caption_id'] for x in range(len(data.keys())) ];
    return caption_ids

  def get_story_captions(self, story_id):
    data =  self._story_index[story_id];
    captions = [ data[x]['caption'] for x in range(len(data.keys())) ];
    return captions

  def get_story_source_data(self, story_id):
    data =  self._story_index[story_id];
    source_datas = [ data[x]['source_data'] for x in range(len(data.keys())) ];
    return source_datas[0];

  @property
  def name(self):
    return self._name
  @property
  def split(self):
    return self._image_set;

  @property
  def coco_caption_data(self):
    return self._coco_caption_data
  
  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def class_to_ind(self):
    return self._class_to_ind
  
  @property
  def image_index(self):
    return self._image_index

  @property
  def num_images(self):
    return len(self.image_index)
