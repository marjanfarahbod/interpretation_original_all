# SAGA annotation automated label interpretation #

This script performs automated interpretation of SAGA annotations. See [this page](http://noble.gs.washington.edu/proj/encyclopedia/) and [this manuscript](http://dx.doi.org/10.1101/086025) for more information.

Required Python packages:

* segtools
* genomedata
* scikit-learn
* pandas
* bedtools-python (https://github.com/arq5x/bedtools-python)
* path (https://pypi.python.org/pypi/forked-path)

Input data:

* Reference annotations. Reference annotations are annotations whose labels have already been interpreted. These interpreted labels will be used as training examples for the classifier. Reference annotations should be in BED4+ format (`chrom\tstart\tend\tlabel`). This script supports using concatenated annotations for training -- that is, several annotations with the same underlying model. It will aggregate features across all cell types and instantiate a single training example for each label, rather than one per label per cell type. Each set of concatenated annotations is associated with a particular "concatenation key", which is a unique identifier for that set of concatenated annotation. If there is no concatenation, give a unique concatenation key to each annotation.
* Target annotations. These are the annotations whose labels you would like to interpret. Same format as reference annotations.
* Reference annotation label mapping. A tab-delimited file with the format `<concatenation key>\t<original label>\t<new label>`. The original label should appear in the reference annotation. The target labels should form a consistent set for all reference annotations, and will be the target of the classifier.
* Signal data for computing histone modification features. The script requires data for six histone modifications (H3K27me3, H3K36me3, H3K4me1, H3K4me3, H3K9me3, H3K27ac) for each cell type which has either a reference or target annotation. This data should be in [genomedata format](https://www.pmgenomics.ca/hoffmanlab/proj/genomedata/). If some cell types are missing this data, it may be appropriate to substitute data from a related cell type (see manuscript).

Arguments:

* `--reference_anns`. A path to a file specifying the reference annotations. Format is `<path>\t<concatenation key>\t<cell type>`.
* `--target_anns`. A path to a file specifying the target annotations. Format is `<path>\t<cell type>`.
* `--label_mappings``. Path to reference annotation label mapping, as described above.
* `--feature_data`. Path to a file specifying location of histone modification signal data. Format is `<cell type>\t<histone modification>\t<genomedata path>\t<name of track within genomedata archive>`.
* `--genes`. A path to a gene annotation in GTF format, to be used for creating gene features in the classifier. A gene annotation can be obtained [from GENCODE](https://www.gencodegenes.org/releases/current.html).
* `--workdir`. All output will go in this directory.

Output: The primary output of this script is label interpretations, which will be placed in `<workdir>/<celltype>/state_mnemonics.txt` . The script will also reproduce several plots from the Libbrecht et al 2016 manuscript.

The script is fully restartable using the same workdir. If you give the script a workdir with partially complete results, it will detect which files are already present and it will not recreate these.

In order to create training features for the annotations (both reference annotations and target annotations), the script will run several segtools analysis scripts on each annotation. This may be quite time-consuming. If you want to speed up this process by, for example, running segtools in parallel on a cluster, you can do this yourself and then copy (or symlink) the segtools output into the appropriate place in the workdir. As with all intermediate files, the script will detect these and will not re-create them.