import _init_paths
import coco_voc
import os
import code
import sg_utils

'''
Simple functions to keep track of datasets and model configurations
'''

__author__ = "Ishan Misra <ishanmisra@gmail.com>"
__date__ = "2016.06.24"

#camelCased functions are "private"

COCO_ROOT_DIR = './data/coco/';
YFCC_ROOT_DIR = '';

def get_vocab_words(vocabFile):
  return [line.strip() for line in open(vocabFile,'r').readlines()];

def get_vocab(vocabFile):
  words = [line.strip() for line in open(vocabFile,'r').readlines()];
  vocab = {};
  for ind, w in enumerate(words):
    vocab[w]=ind;
  return vocab, words;

def getImdbPaths(dataset, splitName):
  imdbPaths = {};
  if dataset == 'coco':
    assert(splitName in ['train', 'val', 'test', 'valid1', 'valid2', 'valid2-1', 'train+val'] );
    assert os.path.exists(COCO_ROOT_DIR);
    imdbPaths['jsonPath'] = os.path.join(COCO_ROOT_DIR, 'annotations','captions_%s2014.json'%(splitName));
    imdbPaths['rootDir'] = COCO_ROOT_DIR;
    imdbPaths['imageDir'] = os.path.join(imdbPaths['rootDir'], 'images')
  elif dataset == 'yfcc':
    assert(splitName in ['train', 'val', 'test'] );
    assert os.path.exists(YFCC_ROOT_DIR);
    imdbPaths['jsonPath'] = os.path.join(YFCC_ROOT_DIR, '/yfcc_%s.json'%(splitName));
    imdbPaths['rootDir'] = YFCC_ROOT_DIR;
    imdbPaths['imageDir'] = os.path.join(imdbPaths['rootDir'], 'images')
  else:
    raise ValueError('dataset not known')
  return imdbPaths

def get_imdb(dataset, splitName):
  imdbPaths = getImdbPaths(dataset, splitName);
  imdb = coco_voc.coco_voc(dataset, splitName, image_path=imdbPaths['imageDir'], captionJsonPath = imdbPaths['jsonPath'])
  return imdb;

def modelVocabConfig():
  solverProtoVocabConfig = {
    #COCO models
    ## MILVC
    'baselines/milCOCO_finetune_solver.prototxt'  : {'vocab': 'coco1k', 'label': 'coco1k',\
     'image_size': 565, 'inference': 'MIL'},
    'baselines/milEnsembCOCO_finetune_solver.prototxt': {'vocab': 'coco1k', 'label': 'coco1k',\
     'image_size': 565, 'inference': 'MIL'},
    'noiseCondImage/milLatentNoiseCOCO_finetune_solver.prototxt':\
     {'vocab': 'coco1k', 'label': 'coco1k', 'image_size': 565, 'inference': 'MILNoise'},
    ## MILVC AlexNet
    'baselines/milAlexCOCO_finetune_solver.prototxt': {'vocab': 'coco1k', 'label': 'coco1k',\
     'image_size': 565, 'inference': 'MIL'},
    'baselines/alexLatentNoiseCOCO_finetune_solver.prototxt':\
     {'vocab': 'coco1k', 'label': 'coco1k', 'image_size': 565, 'inference': 'MILNoise'},

    ## Vanilla classification
    ## we let the inference type remain "MIL" and create appropriate fields in the deploy proto
    ## this makes things easy
    'baselines/classificationCOCO_finetune_solver.prototxt': {'vocab': 'coco1k', 'label': 'coco1k',\
     'image_size': 224, 'inference': 'MIL'},
    'baselines/classificationEnsembCOCO_finetune_solver.prototxt': {'vocab': 'coco1k', 'label': 'coco1k',\
     'image_size': 224, 'inference': 'MIL'},
    'noiseCondImage/classLatetNoiseCOCO_finetune_solver.prototxt':\
     {'vocab': 'coco1k', 'label': 'coco1k', 'image_size': 224},

    #YFCC models
    ## MILVC
    'baselines/milYFCC_finetune_solver.prototxt' : {'vocab': 'yfcc2k', 'label': 'yfcc2k',\
     'image_size': 565, 'inference': 'MIL'},
    'baselines/milEnsembLWYFCC_finetune_solver.prototxt' : {'vocab': 'yfcc2k', 'label': 'yfcc2k',\
     'image_size': 565, 'inference': 'MIL'},
    'noiseCondImage/milLatetNoiseYFCC_finetune_solver.prototxt':\
     {'vocab': 'yfcc2k', 'label': 'yfcc2k', 'image_size': 565, 'inference': 'MILNoise'},
    ## Vanilla classification
    'vgg/classYFCC_finetune_solver.prototxt' : {'vocab': 'yfcc2k', 'label': 'yfcc2k',\
     'image_size': 224, 'inference': 'MIL'},
    'vgg/classYFCCEnsemb_finetune_solver.prototxt' : {'vocab': 'yfcc2k', 'label': 'yfcc2k',\
     'image_size': 224, 'inference': 'MIL'},
    'noiseCondImage/class150YFCCunfzdecay_finetune_solver.prototxt' : {'vocab': 'yfcc2k', 'label': 'yfcc2k',\
     'image_size': 224, 'inference': 'MILNoise'},
  }
  vocabConfig = { \
    'coco1k' : ['vocab_coco', './vocabs/'],\
  }
  labelConfig = { \
    'coco1k' : ['coco1k_%s-%s_label_counts.h5', './data/'],\
  }
  return solverProtoVocabConfig, vocabConfig, labelConfig;

def get_model_vocab_filename(solverProtoKey):
  solverProtoVocabConfig, vocabConfig, labelConfig = modelVocabConfig();
  vocabKey = solverProtoVocabConfig[solverProtoKey]['vocab'];
  vocabOut = vocabConfig[vocabKey];
  vocabFile = os.path.join(vocabOut[1], '%s.pkl'%(vocabOut[0]));
  return vocabFile, vocabKey;

def get_model_vocab(solverProtoKey):
  vocabName, vocabKey = get_model_vocab_filename(solverProtoKey);
  dt = sg_utils.load_variables(vocabName);
  if 'vocab' in dt:
    return dt['vocab'];
  else:
    return dt;

def get_model_inference_type(solverProtoKey):
  solverProtoVocabConfig, vocabConfig, labelConfig = modelVocabConfig();
  return solverProtoVocabConfig[solverProtoKey]['inference'];

def get_model_image_size(solverProtoKey):
  solverProtoVocabConfig, vocabConfig, labelConfig = modelVocabConfig();
  return solverProtoVocabConfig[solverProtoKey]['image_size'];

def get_model_label_filename(solverProtoKey):
  solverProtoVocabConfig, vocabConfig, labelConfig = modelVocabConfig();
  t = labelConfig[solverProtoVocabConfig[solverProtoKey]['label']]
  return os.path.join(t[1], t[0]);