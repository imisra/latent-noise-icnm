import numpy as np


import operator
import functools

def maxk(arr, topK):
  topKInds = arr.argsort()[-topK:]; topKInds = topKInds[::-1];    
  return topKInds;

def mink(arr, topK):
  topKInds = arr.argsort()[:topK];
  return topKInds;

def argsort(seq, reverse=False):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse);

def sublist(l, inds):
    return [l[x] for x in inds];

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1    
