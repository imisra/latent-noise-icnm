import numpy as np
import re
import os

'''
Simple utilities for making life with caffe easier
Includes
- parsing solver without needing to import protobuf
- parsing caffe logs for reading loss/lr values
'''

__author__ = "Ishan Misra <ishanmisra@gmail.com>"
__date__ = "2015.07.12"

def parseSolverProto(solverPath, removePath=True):
  fh = open(solverPath, 'r');
  for line in fh:
    if line.find('train_net:')!=-1 or line.find('net:')!=-1:
      linesp = line.split(':');
      trainProto = linesp[1].strip();
      trainProto = trainProto.replace('"','');
    elif line.find('snapshot_prefix:')!=-1:
      linesp = line.split(':');
      snapshotPrefix = linesp[1].strip();
      snapshotPrefix = snapshotPrefix.replace('"','')+'_iter_';
    elif line.find('snapshot:')!=-1:
      linesp = line.split(':');
      snapshotInterval = linesp[1].strip();
      snapshotInterval = int(snapshotInterval.replace('"',''));
  fh.close();
  if removePath:
    trainProto = os.path.split(trainProto);
    trainProto = trainProto[-1];
    snapshotPrefix = os.path.split(snapshotPrefix);
    snapshotPrefix = snapshotPrefix[-1];

  deployProto = '%s.deploy'%trainProto;
  return trainProto, snapshotPrefix, snapshotInterval, deployProto

def get_log_name_from_solver(solverPath):
  solverName = os.path.split(solverPath);
  solverName = solverName[-1];
  baseName = os.path.splitext(solverName)[0];
  modelName = '%s_%s'%(baseName.split('_')[0], baseName.split('_')[1]);
  logName = '%s_training.log'%(modelName);
  return logName

def get_iter_from_model_file(modelFile):
  modelBase = os.path.basename(modelFile);
  modelBase = os.path.splitext(modelBase)[0];
  modelIter = int(modelBase.split('_')[-1].replace('.caffemodel',''));
  return modelIter;

def get_model_aux_files_from_solver(caffeModelPath = None, expSubDir = None,
  solverName = None, solverProtoPath = None, fullPath = True):
  if solverName == None and solverProtoPath == None:
    return None;
  elif solverProtoPath == None:
    solverProtoPath = os.path.join(caffeModelPath, solverName);
  if caffeModelPath == None:
    caffeModelPath = os.path.split(solverProtoPath)[0];
    caffeModelPath = os.path.split(caffeModelPath)[0]
  if not os.path.isfile(solverProtoPath):
    return None;
  trainProto, snapshotPrefix, \
  snapshotInterval, deployProto = \
    parseSolverProto(solverProtoPath, removePath = True);
  logName = get_log_name_from_solver(solverProtoPath);
  if expSubDir == None:
    expSubDir = solverProtoPath.split(os.path.sep);
    expSubDir = expSubDir[-2];
  expPath = os.path.join(caffeModelPath, expSubDir)
  logPath = os.path.join(expPath, 'logs', logName);

  trainProtoPath =  os.path.join(os.path.split(solverProtoPath)[0], trainProto);
  deployProtoPath =  os.path.join(os.path.split(solverProtoPath)[0], deployProto);
  snapshotDir = os.path.join(expPath, 'cache');
  snapshotPrefixPath = os.path.join(expPath, 'cache', snapshotPrefix);
  cmd = "find %s/ -maxdepth 1 -type f| grep '^%s' | grep 'caffemodel' | sort --version-sort -f"%(snapshotDir, snapshotPrefixPath);
  snapshotFiles = os.popen(cmd).read().split();

  cmd = "find %s/ -maxdepth 1 -type f | grep '^%s' | grep 'solverstate' | sort --version-sort -f"%(snapshotDir, snapshotPrefixPath)
  solverFiles = os.popen(cmd).read().split();

  auxFiles = {};
  auxFiles['logPath'] = logPath;
  auxFiles['expSubDirBase'] = expSubDir;
  auxFiles['expPath'] = expPath;
  auxFiles['deployProtoPath'] = deployProtoPath;
  auxFiles['solverProtoPath'] = solverProtoPath;
  auxFiles['trainProtoPath'] = trainProtoPath;
  auxFiles['snapshotPrefixPath'] = snapshotPrefixPath;
  auxFiles['snapshotFiles'] = snapshotFiles;
  auxFiles['solverFiles'] = solverFiles;
  return auxFiles;

def train_loss_reader(logPath):
  if not os.path.isfile(logPath):
    return None;
  #following - https://docs.python.org/2/library/re.html#simulating-scanf
  iterLossRe = re.compile('Iteration ([-+]?\d+), loss = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)');
  iterLRRe = re.compile('Iteration ([-+]?\d+), lr = ([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)');
  logLines = [str(x.strip()) for x in open(logPath,'r')];
  iterationNumbers = []
  lossVals = []
  lrVals = []
  for l in logLines:
    m = iterLossRe.search(l);
    if m is not None:
      iterationNumbers.append(int(m.groups()[0]));
      lossVals.append(float(m.groups()[1]));
    m = iterLRRe.search(l);
    if m is not None:
      lrVals.append(float(m.groups()[1]));
  return np.array(iterationNumbers), np.array(lossVals), np.array(lrVals)