import _init_paths
import caffe_model_utils as caffe_utils
import caffe
import data_model_utils as meu
import numpy as np
import cv2
import sg_utils
import im_utils
import cap_eval_utils
import test_model as tm
import preprocess
import os
import h5py
import re
import lock_utils
import time
import sys
import code
import traceback as tb
from scipy import stats

__author__ = "Ishan Misra <ishanmisra@gmail.com>"
__date__ = "2016.06.24"

# Simple coding convention for scoping
# Local functions within this file are camelCased. These functions are "private"

#TODO: Add base_image_size as a param
def loadModel(deployProtoPath, modelPath, vocab, base_image_size, infType):
  mean = np.array([[[ 103.939, 116.779, 123.68]]]);
  model = tm.load_model(deployProtoPath, modelPath, base_image_size, mean, vocab);
  model['inf_type'] = infType;
  return model

def getBatchedImList(imdb, model, sendIds=False):
  print 'preparing batchedImList for testing',
  if model['inf_type'] == 'MIL' or model['inf_type'] == 'MILNoise':
    imList = [];
    imBnames, imIds = imdb.get_all_image_bnames_ids();
    imIdList = [];
    for imId in imIds:
      imList.append([imdb.image_path_from_id(imId)]);
      imIdList.append([imId]);
    print 'indiv images'
  if sendIds:
    return imList, imIdList;
  return imList;


def testModelBatch(imdb, model, detection_file):
  if 'words' in model['vocab']:
    N_WORDS = len(model['vocab']['words'])
  else:
    #we are using COCO 80 classes
    N_WORDS = 80;
  batchedImList = getBatchedImList(imdb, model);

  sc = np.zeros((len(batchedImList), N_WORDS), dtype=np.float)
  mil_prob = np.zeros((len(batchedImList), N_WORDS), dtype=np.float)
  if model['inf_type'] == 'MILNoise':
    fields = ['mil', 'mil_max',\
      'qconds10', 'qconds11', 'noisy_comb_noimage']
    qdata_raw = np.zeros((len(batchedImList), 4*N_WORDS), dtype=np.float32)
    qdata_smax = np.zeros((len(batchedImList), 4*N_WORDS), dtype=np.float32)
    qconds10 = np.zeros((len(batchedImList), N_WORDS), dtype=np.float32)
    qconds11 = np.zeros((len(batchedImList), N_WORDS), dtype=np.float32)
    noisy_comb_noimage = np.zeros((len(batchedImList), N_WORDS), dtype=np.float32)

  for bind in range(len(batchedImList)):
    if model['inf_type'] != 'MILNoise':
      mil_prob[bind,:], sc[bind,:] = tm.test_batch(batchedImList[bind], model)
    else:
      fOut = tm.test_batch(batchedImList[bind], model, fields)
      mil_prob[bind,:] = fOut[0];
      sc[bind,:] = fOut[1];
      qconds10[bind,:] = fOut[2];
      qconds11[bind,:] = fOut[3];
      noisy_comb_noimage[bind,:] = fOut[4];
    sg_utils.tic_toc_print(60, 'test_batch : %d/%d (num_per_batch %d)'%(bind, len(batchedImList), len(batchedImList[0])));

  if detection_file is not None and model['inf_type'] != 'MILNoise':
    sg_utils.save_variables(detection_file, [sc, mil_prob], \
    ['sc', 'mil_prob'], overwrite = True)
  elif detection_file is not None:
    sg_utils.save_variables(detection_file, [sc, mil_prob, \
      qconds10, qconds11, noisy_comb_noimage], \
    ['sc', 'mil_prob',\
     'qconds10', 'qconds11',\
     'noisy_comb_noimage'], overwrite = True)

def evalModelBatchOnClassificationCOCOManual(imdb, model, mil_prob, evalFile, cocoFile):
  if 'words' in model['vocab']:
    N_WORDS = len(model['vocab']['words'])
    capt_words = model['vocab']['words'];
    coco2vocab = {};
    with open('./vocabs/coco2vocab_manual_mapping.txt', 'r') as fh:
      for line in fh:
        key = line.split(':')[0].strip();
        coco2vocab[key] = [x.strip() for x in line.split(":")[1].split(',') if len(x.strip()) > 1]
  else:
    capt_words = model['vocab']['allClassNames'];
    N_WORDS = len(capt_words);
    assert(N_WORDS == mil_prob.shape[1]);
    coco2vocab = {};
    with open('./vocabs/coco2vocab_manual_mapping.txt', 'r') as fh:
      for line in fh:
        key = line.split(':')[0].strip();
        foundWords = [x.strip() for x in line.split(":")[1].split(',') if len(x.strip()) > 1]
        if len(foundWords) >= 1:
          coco2vocab[key]=[key];
        else:
          coco2vocab[key]=[];

  cocodt = sg_utils.load(cocoFile);
  cocoLabels = cocodt['gtLabel'];
  cocoMeta = sg_utils.load(cocoFile.replace('.h5','_meta.pkl'));
  coco_cats = cocoMeta['catNameToId'].keys();
  allClassNames = coco_cats;

  num_found = 0; num_images = mil_prob.shape[0];
  indsFound = [];
  for ind, c in enumerate(allClassNames):
    if len(coco2vocab[c]) >= 1:
      num_found+=1;
      indsFound.append(ind);

  classesFound = [allClassNames[ind] for ind in indsFound]
  Ps = np.zeros((num_images, num_found), dtype=np.float32);
  Rs = np.zeros((num_images, num_found), dtype=np.float32);
  aps = np.zeros(num_found, dtype=np.float32);
  scores = np.zeros((num_images, num_found), dtype=np.float32);
  words = []; ctr = 0;
  num_instances = np.zeros(num_found, dtype=np.float32);
  for ind, coco_ind in enumerate(indsFound):
    coco_catid = None;
    vwords = coco2vocab[allClassNames[coco_ind]];
    pred_prob = None;
    for w in vwords:
      vind = capt_words.index(w);
      if pred_prob is None:
        pred_prob = mil_prob[:,vind];
      else:
        pred_prob = np.maximum(pred_prob, mil_prob[:,vind]);
    coco_catid = cocoMeta['catNameToId'][allClassNames[coco_ind]];
    P, R, score, ap = cap_eval_utils.calc_pr_ovr_noref(cocoLabels[:,coco_catid], pred_prob);
    Ps[:,ctr] = P;
    Rs[:,ctr] = R;
    scores[:,ctr] = score
    aps[ctr] = ap;
    num_instances[ctr] = cocoLabels[:,coco_catid].sum();
    words.append(vwords);
    ctr += 1
  sg_utils.save(evalFile, [Ps, Rs, scores, aps, num_instances],\
   ['P', 'R', 'score', 'ap', 'num_instances'], overwrite = True);
  evalMeta = evalFile.replace('.h5','_meta.pkl')
  sg_utils.save(evalMeta,\
   [words, coco2vocab, indsFound, classesFound],\
   ['words', 'coco2vocab', 'indsFound', 'classesFound'], overwrite = True);


def evalModelBatch(imdb, model, gtLabel, numReferencesToEval,\
     detectionFile, evalFile, evalNoiseKey=None):
  N_WORDS = len(model['vocab']['words'])
  vocab = model['vocab']
  imBnames, imIds = imdb.get_all_image_bnames_ids();
  dt = sg_utils.load_variables(detectionFile)
  mil_prob = dt['mil_prob'];

  tm.benchmark_ap(vocab, gtLabel, numReferencesToEval, mil_prob, eval_file = evalFile)
  if evalNoiseKey is not None:
    mil_prob = dt[evalNoiseKey];
    evalNoiseFile = evalFile.replace('.h5','_noise.h5');
    if not lock_utils.isLocked(evalNoiseFile):
      tm.benchmark_ap(vocab, gtLabel, numReferencesToEval, mil_prob, eval_file = evalNoiseFile)
      lock_utils.unlock(evalNoiseFile);

def evalModelBatchNoRef(imdb, model, gtLabel, \
  numReferencesToEval, detectionFile, evalFile, evalNoiseKey=None):
  N_WORDS = len(model['vocab']['words'])
  vocab = model['vocab']
  imBnames, imIds = imdb.get_all_image_bnames_ids();
  gtLabel = np.array(gtLabel > 0, dtype=np.float32);

  dt = sg_utils.load_variables(detectionFile)
  mil_prob = dt['mil_prob'];

  tm.benchmark_only_ap(vocab, gtLabel, numReferencesToEval, mil_prob, eval_file = evalFile, noref = True);
  if evalNoiseKey is not None:
    mil_prob = dt[evalNoiseKey];
    evalNoiseFile = evalFile.replace('.h5','_noise.h5');
    if not lock_utils.is_locked(evalNoiseFile):
      tm.benchmark_only_ap(vocab, gtLabel, numReferencesToEval, mil_prob, eval_file = evalNoiseFile, noref = True)
      lock_utils.unlock(evalNoiseFile);


def getExpNameFromSolverProtoName(solverProtoPath = None):
  solverBase = os.path.basename(solverProtoPath);
  expName = solverBase.replace('_finetune_solver.prototxt','');
  expName = expName.replace('_scratch_solver.prototxt','');
  return expName;

def getModelOutputPaths(detPath, expDirBase, expName, modelFile, testSetName, splitName,\
            numReferencesToEval = None, minWords = None, precThresh = None, ext='.pkl'):
  expOutPath = os.path.join(detPath, expDirBase);
  modelIter = caffe_utils.get_iter_from_model_file(modelFile)
  detectionFile = os.path.join(expOutPath,\
   'detections_%s-%s_%d%s'%(testSetName, splitName, modelIter,ext));
  modelOuts = {};
  modelOuts['detectionFile'] = detectionFile;
  if numReferencesToEval != None:
    evalFile = os.path.join(expOutPath,\
     'eval_%s-%s_%d_ref%d.h5'%(testSetName, splitName, modelIter, numReferencesToEval));
    modelOuts['evalFile'] = evalFile;
  if numReferencesToEval != None and minWords != None and precThresh != None:
    precFile = os.path.join(expOutPath,\
     'prec_%s-%s_%d_ref%d_mw%d_th_%d.txt'%(testSetName, splitName, modelIter, \
                numReferencesToEval, minWords, int(100*precThresh)));
    scFile = os.path.join(expOutPath,\
     'sc_%s-%s_%d_ref%d_mw%d_th_%d.txt'%(testSetName, splitName, modelIter, \
                numReferencesToEval, minWords, int(100*precThresh)));
    modelOuts['precFile'] = precFile;
    modelOuts['scFile'] = scFile;

  modelOuts['expOutPath'] = expOutPath;
  return modelOuts

def printImdbFiles(imdb):
  for i in xrange(len(imdb.image_index)):
    print imdb.image_path_at(i)

def getLabels(imdb, model, solverProtoName):
  labelFileName = meu.get_model_label_filename(solverProtoName);
  labelFileName = labelFileName%(imdb.name, imdb.split);
  if not os.path.exists(labelFileName):
    print 'creating ground truth labels for evaluation'
    imBnames, imIds = imdb.get_all_image_bnames_ids();
    gtLabel = preprocess.get_vocab_counts(imIds, imdb._coco_caption_data, model['vocab']);
    sg_utils.save(labelFileName, [gtLabel], ['gtLabel'], overwrite=True)
  else:
    gtLabel = sg_utils.load(labelFileName)['gtLabel'];
  return gtLabel;

#TODO: Add dataset as param

def mainTest():
##DO NOT CHANGE
  numReferencesToEval = 5;
  minWords = 3;
  precThresh = 0.5;
#####
  testSetName='coco';
  testSetSplit = 'valid2';

  imdb = meu.get_imdb(testSetName, testSetSplit);
  has_gpu = False;

  if has_gpu:
    gpuId = 1
    caffe.set_mode_gpu(); caffe.set_device(gpuId);
  else:
    caffe.set_mode_cpu();
    print 'using CPU'
  #list of paths where we keep our caffe models
  caffeModelPaths = ['./experiments'];
  #output directory to write results
  #make sure it has >2GB free space
  detOutPath = './det-output';
  #list of models we want to evaluate
  #make sure they have an entry in the function modelVocabConfig() in data_model_utils.py
  solverProtoList = [
  'vgg/mil_finetune_solver.prototxt',\
  ]

  #iterations to evaluate
  # evalIters = [80000, 160000, 240000, 320000, 400000];
  evalIters = [320000]
  for i in range(len(solverProtoList)):
    solverProtoName = solverProtoList[i];
    vocab = meu.get_model_vocab(solverProtoName);
    infType = meu.get_model_inference_type(solverProtoList[i]);
    baseImageSize = meu.get_model_image_size(solverProtoList[i]);
    gtKeyedLabel = None

    for caffeModelPath in caffeModelPaths:
      solverProtoPath = os.path.join(caffeModelPath, solverProtoName);
      auxFiles = caffe_utils.get_model_aux_files_from_solver(\
            solverProtoPath = solverProtoPath, caffeModelPath=caffeModelPath);
      if auxFiles == None:
        print 'could not find solver in %s'%(solverProtoPath)
        continue;
      if len(auxFiles['snapshotFiles']) == 0:
        print 'no snapshots found ', solverProtoPath
        continue;
      expSubDirBase = auxFiles['expSubDirBase'];
      expName = getExpNameFromSolverProtoName(solverProtoPath)

      expDirBase = os.path.join(expSubDirBase, expName)
      modelIterNums = [ caffe_utils.get_iter_from_model_file(snapFilePath)\
                       for snapFilePath in auxFiles['snapshotFiles'] ];
      runInds = im_utils.argsort(modelIterNums, reverse=True);
      for ci, s in enumerate(runInds):
        snapFilePath = auxFiles['snapshotFiles'][s];
        modelIterNumber =  caffe_utils.get_iter_from_model_file(snapFilePath);
        if modelIterNumber not in evalIters:
          continue;
        print solverProtoPath, modelIterNumber
        modelOuts = getModelOutputPaths(detOutPath, expDirBase,\
                expName, snapFilePath , testSetName, testSetSplit,\
                numReferencesToEval = numReferencesToEval,
                minWords = minWords, precThresh = precThresh, ext='.h5');
        detectionFile = modelOuts['detectionFile'];
        evalFile = modelOuts['evalFile']; #evaluate as in MILVC
        evalNoRefFile = evalFile.replace('.h5','_noref.h5'); #evaluate using standard definition of AP
        evalCocoManualGtFile = evalFile.replace('.h5','_cocomanualgt.h5'); #evaluate using COCO fully-labeled ground truth
        bdir = os.path.split(detectionFile)[0];
        sg_utils.mkdir_if_missing(bdir);

        if not lock_utils.is_locked(detectionFile):
          model = loadModel(auxFiles['deployProtoPath'], snapFilePath, vocab, baseImageSize, infType);
          testModelBatch(imdb, model, detectionFile);
          lock_utils.unlock(detectionFile);
        else:
          print '%s locked'%(detectionFile)
        model = {};
        model['inf_type'] = infType;
        model['vocab'] = vocab;
        gtLabel = getLabels(imdb, model, solverProtoName);

        #evaluate as in MILVC: using "weighted" version of AP; requires multiple gt references per image
        #e.g. in COCO captions we have 5 captions per image. So we for each "visual concept" we have 5 gt references
        if imdb._name == 'coco' and \
          lock_utils.file_ready_to_read(detectionFile) and (not lock_utils.is_locked(evalFile)):
          model = {};
          model['inf_type'] = infType;
          model['vocab'] = vocab;
          if infType=='MILNoise':
            evalModelBatch(imdb, model, gtLabel, \
              numReferencesToEval, detectionFile, evalFile, evalNoiseKey='noisy_comb_noimage');
          else:
            evalModelBatch(imdb, model, gtLabel,\
             numReferencesToEval, detectionFile, evalFile);
            lock_utils.unlock(evalFile);

        #evaluate using standard AP definition. Does not need multiple references. Hence the name "NoRef"
        if imdb._name == 'coco' and \
          lock_utils.file_ready_to_read(detectionFile) and (not lock_utils.is_locked(evalNoRefFile)):
          model = {};
          model['inf_type'] = infType;
          model['vocab'] = vocab;
          if infType=='MILNoise':
            evalModelBatchNoRef(imdb, model, gtLabel,\
             numReferencesToEval, detectionFile, evalNoRefFile, evalNoiseKey='noisy_comb_noimage');
          else:
            evalModelBatchNoRef(imdb, model, gtLabel,\
             numReferencesToEval, detectionFile, evalNoRefFile);
            lock_utils.unlock(evalNoRefFile);

        #evaluate using fully labeled ground truth from COCO 80 detection classes.
        #we have a manual mapping defined from COCO 80 classes to the 1000 visual concepts
        if imdb._name == 'coco' and \
          lock_utils.file_ready_to_read(detectionFile)\
           and (not lock_utils.is_locked(evalCocoManualGtFile)):
          model = {};
          model['inf_type'] = infType;
          model['vocab'] = vocab;
          cocoFile = './data/coco_instancesGT_eval_%s.h5'%(testSetSplit)
          dt = sg_utils.load(detectionFile);
          mil_prob = dt['mil_prob'];
          evalModelBatchOnClassificationCOCOManual(imdb, model,\
           mil_prob, evalCocoManualGtFile, cocoFile)
          if infType=='MILNoise':
            mil_prob = dt['noisy_comb_noimage'];
            evalCocoManualGtNoiseFile = evalCocoManualGtFile.replace('.h5','_noise.h5')
            evalModelBatchOnClassificationCOCOManual(imdb, model,\
             mil_prob, evalCocoManualGtNoiseFile, cocoFile)
          lock_utils.unlock(evalCocoManualGtFile);

        if imdb.name == 'coco' and lock_utils.file_ready_to_read(evalFile):
          print '=='*20;
          print 'AP (as computed in MILVC)'
          N_WORDS = len(vocab['words'])
          model = {};
          model['inf_type'] = infType;
          model['vocab'] = vocab;
          cap_eval_utils.print_benchmark_latex(evalFile, vocab = vocab);
          evalFile = evalFile.replace('.h5','_noise.h5');
          if os.path.isfile(evalFile):
            print 'noise'
            cap_eval_utils.print_benchmark_latex(evalFile, vocab = vocab);

        if imdb.name == 'coco' and lock_utils.file_ready_to_read(evalNoRefFile):
          print '=='*20;
          print 'AP (as computed in PASCAL VOC)'
          N_WORDS = len(vocab['words'])
          model = {};
          model['inf_type'] = infType;
          model['vocab'] = vocab;
          cap_eval_utils.print_benchmark_latex(evalNoRefFile, vocab = vocab);
          evalNoRefFile = evalNoRefFile.replace('.h5','_noise.h5');
          if os.path.isfile(evalNoRefFile):
            print 'noise'
            cap_eval_utils.print_benchmark_latex(evalNoRefFile, vocab = vocab);

        if imdb.name == 'coco' and lock_utils.file_ready_to_read(evalCocoManualGtFile):
          dt = sg_utils.load(evalCocoManualGtFile);
          dtMeta = sg_utils.load(evalCocoManualGtFile.replace('.h5','_meta.pkl'))
          classesFound = dtMeta['classesFound']
          srtInds = im_utils.argsort(classesFound);
          accAP = np.zeros((1),dtype=np.float32);
          for ind in srtInds:
            accAP += dt['ap'][ind]
          print 'evaluate on fully-labeled GT:',
          print 'AP %.2f; classes %d'%(100*accAP/len(classesFound), len(classesFound))
          evalCocoManualGtNoiseFile = evalCocoManualGtFile.replace('.h5','_noise.h5')
          if os.path.isfile(evalCocoManualGtNoiseFile):
            dt = sg_utils.load(evalCocoManualGtNoiseFile);
            print '--noise--'
            accAP = np.zeros((1),dtype=np.float32);
            for ind in srtInds:
              print '{:.2f} '.format(100*dt['ap'][ind]),
              accAP += dt['ap'][ind];
            print ''
            print '%.2f; %d'%(100*accAP/len(classesFound), len(classesFound))
          print '--'*10

if __name__ == '__main__':
  mainTest();
