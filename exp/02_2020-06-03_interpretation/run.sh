#!/bin/bash
set -x -o errexit -o nounset

workdir=$1

python -i ../01_2020-06-03_files/saga_interpretation/saga_interpretation.py \
	--reference_anns=FIXME \
	--target_anns=FIXME \
	--label_mappings=../01_2020-06-03_files/2016-10-05_label_mappings.txt \
	--feature_data=FIXME \
	--genes=FIXME \
	--workdir=$1



