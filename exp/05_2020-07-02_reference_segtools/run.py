#!/bin/env python
import sys
import os
import argparse
import subprocess
import gzip
import math
import random
from collections import defaultdict
#import bedtools
import shutil
import os

from pathlib import Path

with open("PIP-1210.csv") as f:
    for line in f:
        line = line.split(",")
        if line[0] == "url":
            headers = line
        else:
            url = line[0]
            celltype = line[1]
            concatenation_key = Path(line[4])
            feature_aggregation_tab = line[8]
            signal_distribution_tab = line[9]
            cmd = ["gsutil", "-m", "cp", "-r", feature_aggregation_tab, str(concatenation_key / celltype / "feature_aggregation.tab")]
            print(" ".join(map(str, cmd)))
            subprocess.check_call(cmd)
            cmd = ["gsutil", "-m", "cp", "-r", signal_distribution_tab, str(concatenation_key / celltype / "signal_distribution.tab")]
            print(" ".join(map(str, cmd)))
            subprocess.check_call(cmd)

