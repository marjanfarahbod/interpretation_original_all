#!/bin/env python

##################################
# input args:
# 1. workdir (where the model is pickled and the classifier results created),
# 2. the target dir (where we have list of celltype foldres with segtool files inside)
##################################

##################################
# begin header
import sys
import os
import argparse
import subprocess
import gzip
import math
import random
from path import Path
from collections import defaultdict
#import bedtools
import shutil
import os

sys.path.append(".")
import util 

parser = argparse.ArgumentParser()
parser.add_argument("workdir", type=Path)
parser.add_argument("targetdir", type = Path)
args = parser.parse_args()
workdir = args.workdir
segtooldir = args.targetdir

print(workdir)
print(segtooldir)

