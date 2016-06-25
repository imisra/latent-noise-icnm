import os
import time
'''
Start simple MAP (and hacky REDUCE) processes
- Assumes atomic implementations of os.mkdir and os.rmdir
- Creates "locks" on resources by creating directories in the filesystem
- Multiple processes are synchronizes via filesystem without need for IPC
- Private functions are camelCased.
Example scenario and usage:
- I have a list of 1000 images I want to resize in parallel. In this case
  each image is a resource we can "lock". We can launch multiple python
  processes all of which will run EXACTLY the same code as below.
  (Basically in bash: for i in `seq 10`; do python resize_images.py; done)
-- resize_images.py --
for i in range(1000):
  resize_dst_image = src_image[i].replace('.jpg', '_resized.jpg');
  if lock_utils.is_locked(resize_dst_image):
    continue;
  # do the image resize
  # locking ensures that only one process will resize the image.
  unlock(resize_dst_image)
-----
'''
__author__ = "Ishan Misra <ishanmisra@gmail.com>"
__date__ = "2015.04.20"

def getLockName(filePath):
  lockName = '%s.lock'%(filePath);
  return lockName

def file_ready_to_read(filePath):
  lockName = getLockName(filePath);
  return (not os.path.isdir(lockName)) and os.path.isfile(filePath);

def waitUntilReady(filePath):
  while file_ready_to_read(filePath) == False:
    print 'waiting for',filePath
    time.sleep(2);
  return True;

def lockExists(filePath):
  lockName = getLockName(filePath);
  return os.path.isdir(lockName) or os.path.isfile(filePath)

def is_locked(filePath):
  lockName = getLockName(filePath);
  if not lockExists(filePath):
    try:
      os.mkdir(lockName)
      retVal = False;
    except:
      retVal = True;
  else:
    return True;
  return retVal;

def unlock(filePath):
  lockName = getLockName(filePath);
  try:
    os.rmdir(lockName);
    return True;
  except:
    return False;
