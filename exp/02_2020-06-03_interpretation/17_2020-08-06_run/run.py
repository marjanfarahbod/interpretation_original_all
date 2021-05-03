#!/bin/env python

##################################
# begin header
import sys
import os
import argparse
import subprocess
import gzip
import math
import random
from path import path
from collections import defaultdict
#import bedtools
import shutil
import os

#import util # ~maxwl/util/lib/python/util.py
#import measure_prop # ~maxwl/mp/lib/python/measure_prop.py

parser = argparse.ArgumentParser()
parser.add_argument("workdir", type=path)
args = parser.parse_args()
workdir = args.workdir

if not workdir.exists():
    workdir.makedirs()

shutil.copy(__file__, workdir / "run.py")
experiment_dir = path(os.getcwd())
project_dir = experiment_dir / ".." / ".."
os.chdir(workdir)
workdir = path(".")

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("stdout.txt")
fh.setLevel(logging.DEBUG) # >> this determines the file level
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)# >> this determines the output level
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# end header
##################################

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import numpy
import pandas
import pickle
from copy import deepcopy

import psutil
import resource
process = psutil.Process(os.getpid())
def log_mem():
    psutil_mem = float(process.memory_info().rss) / float(2 ** 30)
    resource_mem = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / float(2 ** 20)
    logger.info("Memory usage -- resource: %.1f gb; psutil: %.1f gb", resource_mem, psutil_mem )

#sys.path.append("/net/gs/vol1/home/oscar01/proj/2013joint-annotations/results/oscar01/2015-06-08/bin/whole_experiment/running_classifier")
#import signal_distribution
#import feature_aggregation_2


#################################################
# Input files
#################################################

#reference_ann_dir = experiment_dir / path("../23_2015-03-24_classified_annotations/")
#reference_anns_list = reference_ann_dir / "nobackup/10_2016-08-31_adding_our_annotations/all_annotations.txt"
reference_anns_list = experiment_dir / "../01_2020-06-03_files/all_annotations.txt"
#segtools_dir = reference_ann_dir / "nobackup/10_2016-08-31_adding_our_annotations/"
segtools_dir = experiment_dir / "../05_2020-07-02_reference_segtools"




#################################################
# For plotting
#################################################

feature_order_R = """c('H3K9me3', 'H3K27me3', 'H3K36me3', 'H3K4me3', 'H3K4me1', 'H3K27ac', "5' flanking (1000-10000 bp)", "5' flanking (1-1000 bp)", 'initial exon', 'initial intron', 'internal exons', 'internal introns', 'terminal exon', 'terminal intron', "3' flanking (1-1000 bp)", "3' flanking (1000-10000 bp)")"""

bio_label_order_R =  """c('Quiescent', 'ConstitutiveHet', 'FacultativeHet', 'Transcribed', 'Promoter', 'Enhancer', 'RegPermissive', 'Bivalent', 'LowConfidence')"""
bio_label_order_ref_R =  """c('Quiescent', 'ConstitutiveHet', 'FacultativeHet', 'Transcribed', 'Promoter', 'Enhancer', 'RegPermissive', 'Bivalent', 'Unclassified')"""


feature_heatmap_disc_vals_R = "c(-Inf, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, Inf)"



#################################################
# Functions for reading segtools directories
#################################################

class SDAnnotation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.signal_distributions = {}
        self.label_names = set()
        self.track_names = set()
        with open(file_path, 'r') as f:
            header = f.readline()
            for line in f:
                line_data = line.split("\t")
                label_name = line_data[0]
                track_name = line_data[1]
                mean = float(line_data[2])
                sd = float(line_data[3])
                n = int(line_data[4])
                self.label_names.add(label_name)
                self.track_names.add(track_name)
                if not label_name in self.signal_distributions:
                    self.signal_distributions[label_name] = {}
                self.signal_distributions[label_name][track_name] = {"mean": mean, "sd": sd, "n": n}
        self.label_names = list(self.label_names)
        self.track_names = list(self.track_names)

def agg_parse_header(header):
    label_name_bases = header.split()
    labels = {}
    label_names = []
    if label_name_bases[0] == "#":
        for label in label_name_bases[1:]:
            label_data = label.split("=")
            label_name = label_data[0]
            if "num_features" == label_name:
                continue
            if "spacers" == label_name:
                continue
            label_bases = int(label_data[1])
            label = AggLabel(label_name, label_bases)
            labels[label_name] = label
            label_names.append(label_name)
    return labels, label_names

def agg_parse(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().rstrip()
        labels, label_names = agg_parse_header(header)
        second_header = f.readline()
        label_names = second_header.rstrip().split("\t")[3:]
        for line in f:
            line_data = line.rstrip().split("\t")
            group = line_data[0]
            component = line_data[1]
            offset = line_data[2]
            label_counts = line_data[3:]
            assert len(label_counts) == len(label_names)
            #label_names = [ int(label_name) for label_name in label_names] # sort to match counts
            #label_names.sort() # sort to match counts
            #label_names = [ str(label_name) for label_name in label_names] # sort to match counts
            for label_name, label_count in zip(label_names, label_counts):
                labels[label_name].add_raw_count(component,int(label_count))
    return labels, label_names

class AggLabel:
    def __init__(self, label_name, label_bases):
        self.label_name = label_name
        self.label_bases = label_bases
        self.component_raw_counts = {}
        self.component_enrichment = {}
    def add_raw_count(self, component, label_count):
        if component not in self.component_raw_counts:
            self.component_raw_counts[component] = []
        self.component_raw_counts[component].append(label_count)
    def num_bases(self):
        return self.label_bases
    def raw_counts(self):
        return self.component_raw_counts
    def set_enrichment(self,component_enrichment):
        self.component_enrichment = component_enrichment
    def enrichment(self,*args):
        if len(args) == 0:
            return sum([self.component_enrichment[component] for component in self.component_enrichment],[])
        return self.component_enrichment[args[0]]
    def raw_enrichment(self,component):
        return self.component_raw_counts[component]
    def name(self):
        return self.label_name
    def component_enrichments(self):
        return self.component_enrichment

class AggAnnotation:
    def __init__(self, file_path):
        self.file_path = file_path
        self.labels, self.label_names = agg_parse(file_path)
        self.set_enrichment()
    def __iter__(self):
        return iter(self.labels.values())
    def set_enrichment(self):
        genome_bases = 0.0
        component_sum_counts = {}
        for label in self.labels:
            genome_bases += self.labels[label].num_bases()
            component_raw_counts = self.labels[label].raw_counts()
            for component in component_raw_counts:
                if component not in component_sum_counts:
                    component_sum_counts[component] = numpy.zeros(len(component_raw_counts[component]))
                component_sum_counts[component] += numpy.array(component_raw_counts[component])
        for label in self.labels:
            component_raw_counts = self.labels[label].raw_counts()
            component_enrichment = {}
            for component in component_raw_counts:
                if component not in component_enrichment:
                    component_enrichment[component] = []
                for raw_count, sum_count in zip(component_raw_counts[component],component_sum_counts[component]):
                    f_obs = ((raw_count + 1)/ sum_count)
                    f_rand = (self.labels[label].num_bases() / genome_bases)
                    enr  = math.log((f_obs/f_rand),2)
                    component_enrichment[component].append(enr)
            self.labels[label].set_enrichment(component_enrichment)
    def enrichment(self,label_name,*args):
        if len(args) == 0:
            return self.labels[label_name].enrichment()
        assert len(args) == 1
        if args[0] == "components":
            return self.labels[label_name].component_enrichments()
        else:
            return self.labels[label_name].enrichment(args[0])
    def raw_enrichment(self,label_name,component):
        return self.labels[label_name].raw_enrichment(component)
    def get_labels(self):
        return self.labels
    def get_label_names(self):
        return self.label_names



#########################################
# Get list of reference annotations
#########################################
logger.info("Getting list of reference annotations...")
log_mem()

reference_anns = []
reference_concats = {}
histone_features = ["H3K27ME3","H3K36ME3","H3K4ME1","H3K4ME3","H3K9ME3","H3K27AC"]
with open(reference_anns_list, "r") as f:
    for line in f:
        line = line.split()
        if line[0] == "url": continue
        celltype = line[1]
        if celltype == "HELA-S3": continue # data is missing for HELA for some reason
        concatenation_key = line[4]
        # FIXME read signal_dist_fname and agg_fname from reference_anns_list
        signal_dist_fname = segtools_dir / concatenation_key / celltype / "signal_distribution.tab"
        gene_agg_fname = segtools_dir / concatenation_key / celltype /  "feature_aggregation.tab"
        ann = {"url": line[0], "celltype": celltype, "tool": line[2], "dataset_key": line[3], 
                "concatenation_key": concatenation_key, "assembly": line[5], 
                "signal_dist_fname": signal_dist_fname, "gene_agg_fname": gene_agg_fname}
        #if concatenation_key == "segway_nar_H1-HESC":
            #continue # don't have data for this one somehow
        #missing_segtools = False
        #gencode_path = segtools_dir / concatenation_key / celltype / "aggregation/GENCODE/feature_aggregation.tab"
        #if not gencode_path.exists():
            #missing_segtools = True
            #logger.error("Missing {gencode_path}".format(**locals()))
            #raise Exception("Missing {gencode_path}".format(**locals()))
        #for histone in histone_features:
            #histone_path = segtools_dir / concatenation_key / celltype / "signal_distribution/HISTONE.{histone}/signal_distribution.tab".format(**locals())
            #if not histone_path.exists():
                #logger.error("Missing {histone_path}".format(**locals()))
                #raise Exception("Missing {histone_path}".format(**locals()))
                #missing_segtools = True
        #if missing_segtools:
            #logger.warning("!! Skipping because missing segtools data: {ann}".format(**locals()))
            ##raise Exception("!! Skipping because missing segtools data: {ann}".format(**locals()))
            #continue
        reference_anns.append(ann)
        if not (concatenation_key in reference_concats):
            reference_concats[concatenation_key] = []
        reference_concats[concatenation_key].append(ann)

#########################################
# Make classifier data tab
#########################################

log_mem()

def labels_from_segtools_dir(gencode_path):
    #gencode_path = summary_dirpath / "aggregation/GENCODE/feature_aggregation.tab"
    #gencode_path = summary_dirpath / "feature_aggregation.tab"
    gencode = AggAnnotation(gencode_path)
    labels = gencode.get_labels()
    return labels


def feature_name_map(feature_name):
    if feature_name == "H3K9ME3": return "(01) H3K9me3"
    if feature_name == "H3K27ME3": return "(02) H3K27me3"
    if feature_name == "H3K36ME3": return "(03) H3K36me3"
    if feature_name == "H3K4ME3":  return "(04) H3K4me3"
    if feature_name == "H3K27AC": return "(05) H3K27ac"
    if feature_name == "H3K4ME1":  return "(06) H3K4me1"
    if feature_name == "5' flanking (1000-10000 bp)":  return "(07) 5' flanking (1000-10000 bp)"
    if feature_name == "5' flanking (1-1000 bp)": return "(08) 5' flanking (1-1000 bp)"
    if feature_name.startswith("initial exon"): return "(09) initial exon"
    if feature_name.startswith("initial intron"): return "(10) initial intron"
    if feature_name.startswith("internal exons"): return "(11) internal exons"
    if feature_name.startswith("internal introns"): return "(12) internal introns"
    if feature_name.startswith("terminal exon"): return "(13) terminal exon"
    if feature_name.startswith("terminal intron"): return "(14) terminal intron"
    if feature_name == "3' flanking (1-1000 bp)": return "(15) 3' flanking (1-1000 bp)"
    if feature_name == "3' flanking (1000-10000 bp)": return "(16) 3' flanking (1000-10000 bp)"
    else:
        raise Exception("Unrecognized feature name {}".format(feature_name))

#def features_from_segtools_dir(summary_dirpath):
def features_from_segtools_dir(gencode_path, histone_path):
    feature_names = set()
    ann_features = {} # {label: {feature_name: val} }
    ann_label_bases = {}
    celltype = ann["celltype"]
    dataset_key = ann["dataset_key"]
    concatenation_key = ann["concatenation_key"]
    #gencode_path = summary_dirpath / "aggregation/GENCODE/feature_aggregation.tab"
    #gencode_path = summary_dirpath / "feature_aggregation.tab"
    gencode_labels = set()
    gencode = AggAnnotation(gencode_path)
    for label, gencode_label_info in gencode.labels.items():
        gencode_labels.add(label)
        if not (label in ann_features):
            ann_features[label] = {}
        ann_label_bases[label] = gencode_label_info.num_bases()
        for component_name, component_enrichments in gencode_label_info.component_enrichments().items():
            if ("UTR" in component_name) or ("CDS" in component_name): continue
            # Split 5' and 3' flanking regions into two parts. 
            if component_name.startswith("5' flanking"):
                feature_name = "5' flanking (1-1000 bp)"
                feature_name = feature_name_map(feature_name)
                if len(component_enrichments) == 50:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[-5:])
                else:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[-1000:])
                feature_names.add(feature_name)
                feature_name = "5' flanking (1000-10000 bp)"
                feature_name = feature_name_map(feature_name)
                if len(component_enrichments) == 50:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[:-5])
                else:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[:-1000])
                feature_names.add(feature_name)
            elif component_name.startswith("3' flanking"):
                feature_name = "3' flanking (1-1000 bp)"
                feature_name = feature_name_map(feature_name)
                if len(component_enrichments) == 50:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[:5])
                else:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[:1000])
                feature_names.add(feature_name)
                feature_name = "3' flanking (1000-10000 bp)"
                feature_name = feature_name_map(feature_name)
                if len(component_enrichments) == 50:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[5:])
                else:
                    ann_features[label][feature_name] = numpy.mean(component_enrichments[1000:])
                feature_names.add(feature_name)
            else:
                feature_name = feature_name_map(component_name)
                ann_features[label][feature_name] = numpy.mean(component_enrichments)
                feature_names.add(feature_name)
    #for histone in histone_features:
        #histone_path = summary_dirpath / "signal_distribution/HISTONE.{histone}/signal_distribution.tab".format(**locals())
    #histone_path = summary_dirpath / "signal_distribution.tab".format(**locals())
    #if not histone_path.exists():
        #logger.warning("!!! Missing {histone_path}!!".format(**locals()))
        #raise Exception("!!! Missing {histone_path}!!".format(**locals()))
        #for label in ann_features:
            #feature_name = histone
            #feature_name = feature_name_map(feature_name)
            #ann_features[label][feature_name] = float("nan")
            #feature_names.add(feature_name)
    #else:
    histone_signal = SDAnnotation(histone_path) 
    assert(len(histone_signal.track_names) == 6)
    print histone_signal.track_names
    signal_labels = set()
    for label_name in histone_signal.label_names:
        signal_labels.add(label_name)
        for i,track_name in enumerate(histone_signal.track_names):
            # FIXME ! Use tab file with mapping from tracknames to histone feature names
            histone_features = [ "H3K9ME3", "H3K27ME3", "H3K36ME3", "H3K4ME3", "H3K27AC", "H3K4ME1" ]
            feature_name = histone_features[i]
            feature_name = feature_name_map(feature_name)
            ann_features[label_name][feature_name] = histone_signal.signal_distributions[label_name][track_name]["mean"]
            feature_names.add(feature_name)
    if not signal_labels == gencode_labels:
        raise Exception("signal_labels and gencode_labels don't match: signal_labels = {}. gencode_labels = {}".format(signal_labels, gencode_labels))
    return ann_features, ann_label_bases, feature_names


classifier_tab_fname = path("classifier_data.tab")
if classifier_tab_fname.exists():
    logger.info("Classifier tab data exists, reading from file.")
    log_mem()
else:
    logger.info("Creating classifier tab data...")
    log_mem()
    classifier_data = [] # [{bio_label: bio_label, orig_label: orig_label features: {feature_name: val}}]
    feature_names = set()
    for concatenation_key in reference_concats:
        logger.info("Starting concatenation_key: {concatenation_key}".format(**locals()))
        concat_labels = set()
        for ann in reference_concats[concatenation_key]:
            celltype = ann["celltype"]
            #dataset_key = ann["dataset_key"]
            concatenation_key = ann["concatenation_key"]
            #gencode_path = segtools_dir / concatenation_key / celltype / "aggregation/GENCODE/feature_aggregation.tab"
            #gencode = AggAnnotation(gencode_path)
            #summary_dirpath = segtools_dir / concatenation_key / celltype
            labels = labels_from_segtools_dir(ann["gene_agg_fname"])
            concat_labels = concat_labels.union(set(labels))
        logger.info("Found labels: {concat_labels}".format(**locals()))
        concat_features = {label: {} for label in concat_labels}
        concat_label_bases = {label: 0 for label in concat_labels}
        for ann in reference_concats[concatenation_key]:
            celltype = ann["celltype"]
            dataset_key = ann["dataset_key"]
            concatenation_key = ann["concatenation_key"]
            summary_dirpath = segtools_dir / concatenation_key / celltype
            ann_features, ann_label_bases, ann_feature_names = features_from_segtools_dir(ann["gene_agg_fname"], ann["signal_dist_fname"])
            for label,features in ann_features.items():
                assert(len(features) == 16)
            feature_names = feature_names.union(ann_feature_names)
            for label in ann_features:
                for feature_name in ann_features[label]:
                    if feature_name in concat_features[label]:
                        feature_val = ((concat_features[label][feature_name]*concat_label_bases[label]
                                                                 + ann_features[label][feature_name]*ann_label_bases[label])
                                                                / (concat_label_bases[label] + ann_label_bases[label]))
                        assert numpy.isfinite(feature_val)
                        concat_features[label][feature_name] = feature_val
                    else:
                        concat_features[label][feature_name] = ann_features[label][feature_name]
                concat_label_bases[label] += ann_label_bases[label]
        for label, features in concat_features.items():
            #bio_label = label_mappings[concatenation_key][label]
            classifier_data.append({"orig_label": label, "concatenation_key": concatenation_key, "features": features})
    feature_names = list(feature_names)
    with open(classifier_tab_fname, "w") as f:
        f.write("concatenation_key\t")
        #f.write("label\t")
        f.write("orig_label\t")
        f.write("\t".join(feature_names))
        f.write("\n")
        for example in classifier_data:
            f.write(example["concatenation_key"])
            f.write("\t")
            #f.write(example["bio_label"])
            #f.write("\t")
            f.write(example["orig_label"])
            for feature_name in feature_names:
                f.write("\t")
                f.write(str(example["features"][feature_name]))
            f.write("\n")

#########################################
# Get bio label conversions
#########################################
log_mem()
label_mappings = {}
labels_set = set()
orig_labels_set = set()
label_mappings_fname = experiment_dir / "label_mappings.txt"
label_mappings_fname.copy("label_mappings.txt")
with open("label_mappings.txt", "r") as f:
    for line in f:
        line = line.split()
        if line[0] == "concatenation_key":
            continue
        concatenation_key = line[0]
        orig_label = line[1]
        label = line[2]
        if not concatenation_key in label_mappings:
            label_mappings[concatenation_key] = {}
        label_mappings[concatenation_key][orig_label] = label
        labels_set.add(label)
        orig_labels_set.add(orig_label)

all_ref_bio_labels = set.union(*map(set, map(lambda x: x.values(), label_mappings.values())))

logger.info("label_mappings: {label_mappings}".format(**locals()))
log_mem()


#######################################################
# Plot training data
#######################################################

#path("reference").rmtree() # XXX
if not path("reference").exists():
    path("reference").makedirs()
    key = "reference/features"
    script_fname = workdir / "{key}.R".format(**locals())
    script = \
"""
require(Cairo)
require(reshape2)
require(ggplot2)
require(RColorBrewer)

data_fname <- "{classifier_tab_fname}"
data <- read.delim(data_fname, header=TRUE, check.names=FALSE)

for (col in names(data)[3:length(names(data))]) {{
    data[,col] <- data[,col] - mean(data[,col])
    data[,col] <- data[,col] / sd(data[,col])
}}

label_mapping_fname <- "label_mappings.txt"
label_mapping <- read.delim(label_mapping_fname, header=TRUE)
for (i in 1:nrow(label_mapping)) {{
    concatenation_key <- label_mapping[i, "concatenation_key"]
    orig_label <- label_mapping[i, "orig_label"]
    bio_label <- label_mapping[i, "new_label"]
    mask <- (as.character(data$concatenation_key) == concatenation_key) & (as.character(data$orig_label) == orig_label)
    data[mask, "bio_label"] <- bio_label
}}

for (i in 1:nrow(data)) {{
    data[i, "label_str"] <- paste(data[i, "bio_label"], data[i, "orig_label"], data[i, "concatenation_key"], sep="-")
}}

melt_data <- melt(data, id.vars=c("concatenation_key", "orig_label", "bio_label", "label_str"))

melt_data$variable <- factor(melt_data$variable, levels = sort(levels(melt_data$variable)))
melt_data$value_disc <- cut(melt_data$value, breaks={feature_heatmap_disc_vals_R})
palette="RdBu"
colors <- brewer.pal(palette, n=length(levels(melt_data$value_disc)))
color_mapping <- levels(melt_data$value_disc)
colors <- rev(colors)

p <- ggplot(melt_data) +
  aes(x=value) +
  stat_ecdf() +
  scale_x_continuous(breaks=seq(-2,5,0.5), limits=c(-3,5)) +
  scale_y_continuous(breaks=seq(0,1,0.1)) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(paste("{key}_ecdf.pdf", sep=""), p, width=10, height=6, units="in")

for (bio_label in levels(melt_data$bio_label)) {{
    bio_label_data <-  melt_data[melt_data$bio_label == bio_label,]
    if (nrow(bio_label_data) >= 30) {{ # XXX
        cast_data <- acast(bio_label_data, label_str ~ variable, value.var="value")
        ord <- hclust( dist(cast_data, method = "euclidean") )$order
        bio_label_data$label_str <- ordered(bio_label_data$label_str, levels=rownames(cast_data)[ord])

        plot_height <- 3 + 0.0125*nrow(bio_label_data)
        p <- ggplot(bio_label_data) +
          aes(x=variable, y=label_str, fill=value_disc) +
          geom_tile() +
          scale_fill_manual(values=colors, breaks=color_mapping, drop=FALSE) +
          theme_bw() +
          theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
        ggsave(paste("{key}_bylabel_", bio_label, ".pdf", sep=""), p, width=9, height=plot_height, units="in")
    }}
}}

for (concatenation_key in levels(melt_data$concatenation_key)) {{
    if (nrow(melt_data[melt_data$concatenation_key == concatenation_key,]) > 0) {{
        plot_height <- 3 + 0.0125*nrow(melt_data[melt_data$concatenation_key == concatenation_key,])
        p <- ggplot(melt_data[melt_data$concatenation_key == concatenation_key,]) +
          aes(y=variable, x=label_str, fill=value_disc) +
          geom_tile() +
          scale_fill_manual(values=colors, breaks=color_mapping, drop=FALSE) +
          theme_bw() +
          theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
        ggsave(paste("{key}_byann_", concatenation_key, ".pdf", sep=""), p, width=9, height=plot_height, units="in")
    }}
}}

cast_data <- acast(melt_data, label_str ~ variable, value.var="value")
ord <- hclust( dist(cast_data, method = "euclidean") )$order
melt_data$label_str <- ordered(melt_data$label_str, levels=rownames(cast_data)[ord])
plot_height <- 3 + 0.0125*nrow(melt_data)
p <- ggplot(melt_data) +
  aes(x=variable, y=label_str, fill=value_disc) +
  geom_tile() +
  scale_fill_manual(values=colors, breaks=color_mapping, drop=FALSE) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(paste("{key}_all.pdf", sep=""), p, width=9, height=plot_height, units="in", limitsize=FALSE)

    """.format(**locals())
    with open(script_fname, "w") as f:
        f.write(script)
    cmd = ["Rscript", script_fname]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

#######################################################
# Convert classifier data to numpy matrix
#######################################################

log_mem()

classifier_data_frame = pandas.read_csv(classifier_tab_fname, sep="\t")
example_bio_labels = [None for i in range(classifier_data_frame.shape[0])]
for i in range(classifier_data_frame.shape[0]):
    concatenation_key = classifier_data_frame.concatenation_key[i]
    orig_label = classifier_data_frame.orig_label[i]
    if orig_label in label_mappings[concatenation_key]:
        example_bio_labels[i] = label_mappings[concatenation_key][orig_label]
    else:
        logger.warning("No label mapping for {concatenation_key} {orig_label}".format(**locals()))
        example_bio_labels[i] = "??"

example_bio_labels = numpy.array(example_bio_labels)
feature_names = numpy.array(classifier_data_frame.drop("orig_label", 1).drop("concatenation_key",1).columns)
example_orig_labels = numpy.array(classifier_data_frame.orig_label)

# compute feature means/stdevs
feature_means = {}
feature_stdevs = {}
for feature_name in feature_names:
    feature_means[feature_name] = numpy.mean(classifier_data_frame[feature_name])
    feature_stdevs[feature_name] = numpy.std(classifier_data_frame[feature_name])

def features_frame_to_matrix(features_frame, feature_names):
    num_examples = features_frame.shape[0]
    mat = numpy.empty(shape=(num_examples, len(feature_names)), dtype="float")
    for feature_index, feature_name in enumerate(feature_names):
        for example_index in range(num_examples):
            feature_val = features_frame.loc[example_index, feature_name]
            norm_feature_val = float(feature_val - feature_means[feature_name]) / feature_stdevs[feature_name]
            mat[example_index, feature_index] = norm_feature_val
    return mat

#example_features = numpy.array(classifier_data_frame.drop("orig_label", 1).drop("concatenation_key",1))
example_features = features_frame_to_matrix(classifier_data_frame, feature_names)

#features_fname = path("features.npy")
#labels_fname = path("labels.npy")
#orig_labels_fname = path("orig_labels.npy")
#feature_names_fname = path("feature_names.npy")
#if features_fname.exists():
    #logger.info("Training data already exists, reading from file...")
    #features = numpy.load(features_fname)
    #labels = numpy.load(labels_fname)
    #orig_labels = numpy.load(orig_labels_fname)
    #feature_names = numpy.load(feature_names_fname)
#else:
    #logger.info("Starting processing training data...")
    #features = []
    #labels = []
    #orig_labels = []
    #with open(classifier_tab_fname, 'r') as f:
        #header = f.readline().split("\t")
        #feature_names = header[3:]
        #for line in f:
            #line_data = line.rstrip().split('\t')
            #annotation = line_data[0]
            #labels.append(line_data[1])
            #orig_labels.append(line_data[2])
            #features.append(numpy.array(line_data[3:], dtype="float"))
    #features = numpy.vstack(features)
    #labels = numpy.array(labels)
    #orig_labels = numpy.array(orig_labels)
    #numpy.save(features_fname, features)
    #numpy.save(labels_fname, labels)
    #numpy.save(orig_labels_fname, orig_labels)
    #numpy.save(feature_names_fname, feature_names)

#######################################################
# Use cross-validation to compute accuracy
#######################################################

log_mem()

def make_model(reg):
    return RandomForestClassifier(min_samples_leaf=reg, criterion="entropy")
    #return DecisionTreeClassifier(min_samples_leaf=reg, criterion="entropy")
    #return DecisionTreeClassifier(min_samples_split=reg)
    #return DecisionTreeClassifier(min_samples_split=reg, class_weight="balanced")
    #return SGDClassifier(loss="log", penalty="elasticnet", l1_ratio=0.5, n_iter=200, alpha=reg)
    #return LogisticRegression(penalty="l1",C=reg)

xval_dir = path("xval")
scores_outfn = xval_dir / path("scores.tab")
confusion_outfn = xval_dir / "confusion.tab"
orig_label_assignments_outfn = xval_dir / "orig_label_assignments.tab"
random.seed("1989-08-05")
model_ignore_labels = ["??"]
ignore_label_mask = numpy.zeros(len(example_bio_labels), dtype="bool")
for ignore_label in model_ignore_labels:
    ignore_label_mask |= example_bio_labels == ignore_label
model_labels = example_bio_labels[numpy.logical_not(ignore_label_mask)]
model_orig_labels = example_orig_labels[numpy.logical_not(ignore_label_mask)]
model_features = example_features[numpy.logical_not(ignore_label_mask),:]
num_model_examples = len(model_labels)
num_xval_folds = num_model_examples
order = random.sample(range(num_model_examples), num_model_examples)
if not xval_dir.exists():
    xval_dir.makedirs()
if scores_outfn.exists():
    logger.info("scores.tab exists, skipping regularization parameter grid search.")
    log_mem()
else:
    logger.info("Starting regularization parameter grid search")
    log_mem()
    tmp_scores_outfn = xval_dir / path("scores_tmp.tab")
    scores_out = open(tmp_scores_outfn,"w")
    scores_out.write("fold\treg\taccuracy\n")
    confusion_out = open(confusion_outfn, "w")
    confusion_out.write("reg\ttrue_label\tpredicted_label\tprob\tassignments\n")
    orig_label_assignments_f = open(orig_label_assignments_outfn, "w")
    #for reg in numpy.exp2(numpy.arange(-12.0,3.0,0.5)):
    #for reg_index, reg in enumerate(numpy.power(10, numpy.arange(-8.0,1.0,0.5))):
    for reg_index, reg in enumerate([1,2,4,6,8,10,12,16,20,24,28,32,64]):
        logger.info("Starting regularization parameter {0}".format(reg))
        confusion_assignments = {}
        confusion_probability = {}
        orig_label_assignments_key = xval_dir / "orig_label_assignments_{reg}".format(**locals())
        for xval_fold in range(num_xval_folds):
            test_fold_start = int(xval_fold * num_model_examples * float(1)/num_xval_folds)
            test_fold_end = int((xval_fold+1) * num_model_examples * float(1)/num_xval_folds)
            test_fold_indices = order[test_fold_start:test_fold_end]
            train_fold_indices = list(set(range(num_model_examples)) - set(test_fold_indices))
            train_features = model_features[train_fold_indices,:]
            train_labels = model_labels[train_fold_indices]
            test_features = model_features[test_fold_indices,:]
            test_labels = model_labels[test_fold_indices]
            test_orig_labels = model_orig_labels[test_fold_indices]
            model = make_model(reg)
            model.fit(train_features,train_labels)
            score = model.score(test_features,test_labels)
            #logger.info("xval_fold: {xval_fold} ; reg: {reg} ; score: {score}".format(**locals()))
            scores_out.write("%s\t%s\t%s\n" % (xval_fold, reg, score))
            if (xval_fold == 0) and (reg_index == 0):
                labels_str = "\t".join(model.classes_)
                orig_label_assignments_f.write("reg\torig_label\ttrue_label\tpredicted_label\t{labels_str}\n".format(**locals()))
            log_probs = model.predict_log_proba(test_features)
            probs = model.predict_proba(test_features)
            predictions = model.predict(test_features)
            for example_index in range(log_probs.shape[0]):
                true_label = test_labels[example_index]
                orig_label = test_orig_labels[example_index]
                predicted_label = predictions[example_index]
                probs_str = "\t".join(map(str, probs[example_index,:]))
                orig_label_assignments_f.write("{reg}\t{orig_label}\t{true_label}\t{predicted_label}\t{probs_str}\n".format(**locals()))
                if not true_label in confusion_probability: confusion_probability[true_label] = {}
                if not true_label in confusion_assignments: confusion_assignments[true_label] = {}
                if not predicted_label in confusion_assignments[true_label]: confusion_assignments[true_label][predicted_label] = 0
                confusion_assignments[true_label][predicted_label] += 1
                for label_index in range(log_probs.shape[1]):
                    predicted_label = model.classes_[label_index]
                    if not predicted_label in confusion_probability[true_label]: confusion_probability[true_label][predicted_label] = 0
                    if not predicted_label in confusion_assignments[true_label]: confusion_assignments[true_label][predicted_label] = 0
                    confusion_probability[true_label][predicted_label] += log_probs[example_index, label_index]

        for true_label in confusion_probability:
            for predicted_label in confusion_probability[true_label]:
                prob = confusion_probability[true_label][predicted_label]
                assignments = confusion_assignments[true_label][predicted_label]
                confusion_out.write("{reg}\t{true_label}\t{predicted_label}\t{prob}\t{assignments}\n".format(**locals()))
    confusion_out.close()
    scores_out.close()
    orig_label_assignments_f.close()
    tmp_scores_outfn.move(scores_outfn)
    key = xval_dir / "scores"
    data_fname = workdir / "{key}.tab".format(**locals())
    script_fname = workdir / "{key}.R".format(**locals())
    script = \
"""
require(Cairo)
require(ggplot2)
#require(RColorBrewer)

data <- read.delim("{data_fname}", header=TRUE)

p <- ggplot(data) +
  aes_string(x="reg", y="accuracy") +
  stat_summary(fun.data="mean_cl_boot") +
  scale_x_log10(name="Regularization") +
  scale_y_continuous(name="Accuracy") +
  theme_bw()
ggsave("{key}.pdf", p, width=4, height=3, units="in")
""".format(**locals())
    with open(script_fname, "w") as f:
        f.write(script)
    cmd = ["Rscript", script_fname]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)
if True:
    key = xval_dir / "confusion".format(**locals())
    data_fname = confusion_outfn
    script_fname = workdir / "{key}.R".format(**locals())
    script = \
"""
require(Cairo)
require(ggplot2)
require(RColorBrewer)

data <- read.delim("{data_fname}", header=TRUE)
data$true_label <- as.factor(data$true_label)
data$predicted_label <- as.factor(data$predicted_label)
data$true_label <- ordered(data$true_label, {bio_label_order_ref_R})
data$predicted_label <- ordered(data$predicted_label, {bio_label_order_ref_R})
data$reg <- as.factor(data$reg)

for (true_label in levels(data$true_label)) {{
    data[data$true_label == true_label, "assignment_frequency"] <- data[data$true_label == true_label, "assignments"] / (sum(data[data$true_label == true_label, "assignments"]) / length(levels(data$reg)))
}}

data$assignment_frequency_disc <- cut(data$assignment_frequency, breaks=c(-Inf, 0, 0.1, 0.4, 1))
palette="Reds"
colors <- brewer.pal(palette, n=length(levels(data$assignment_frequency_disc)))
color_mapping <- levels(data$assignment_frequency_disc)

for (reg in levels(data$reg)) {{
    p <- ggplot(data[data$reg == reg, ], aes(x=true_label, y=predicted_label, fill=assignment_frequency_disc, label=assignments)) +
      geom_tile() +
      geom_text(size=4) +
      scale_fill_manual(values=colors, name="Frequency", breaks=color_mapping, drop=FALSE) +
      scale_x_discrete(name="True label") +
      scale_y_discrete(name="Predicted label") +
      coord_fixed() +
      theme_classic() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
    ggsave(paste("{key}_", reg, ".pdf", sep=""), p, width=5, height=4, units="in")
}}
""".format(**locals())
    with open(script_fname, "w") as f:
        f.write(script)
    cmd = ["Rscript", script_fname]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)


#######################################################
# Train final model
#######################################################

logger.info("Training final model...")
log_mem()
#reg = 1e-2
reg = 10
model = make_model(reg)
model.fit(model_features, model_labels)

with gzip.open("model.pickle.gz", "w") as f:
    pickle.dump(model, f)

# Plot tree for DT model
if model.__class__ == DecisionTreeClassifier:
    logger.info("Making decision tree plot...")
    if not path("model").exists():
        path("model").makedirs()
    export_graphviz(model,
                    out_file="model/tree_model_orig.dot",
                    feature_names=feature_names)
    with open("model/tree_model_classes.dot", "w") as outf:
        with open("model/tree_model_orig.dot", "r") as inf:
            for line in inf:
                if "value" in line:
                    line = line.split("\\n")
                    for part in line:
                        if part.startswith("value"):
                            part = part.split(",")
                            class_data = map(float, part[0].split("[")[1].split("]")[0].split())
                            class_data_sorted = sorted([(class_index, model.classes_[class_index], class_data[class_index]) for class_index in range(len(class_data))], key = lambda x: x[2], reverse=True)
                            first_entry = True
                            for class_index, class_name, num_examples in class_data_sorted:
                                num_examples = int(num_examples)
                                if num_examples > 0:
                                    if first_entry:
                                        first_entry = False
                                    else:
                                        outf.write("; ")
                                    outf.write("{class_name}: {num_examples}".format(**locals()))
                            outf.write("\"")
                            outf.write(part[1]) # shape=box \n
                        else:
                            outf.write(part)
                            outf.write("\\n")
                else:
                    outf.write(line)
    cmd = ["/usr/bin/dot", "-Tpng",
           "model/tree_model_classes.dot",
           "-o", "model/tree_model_classes.png" ]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)

# Plot coefs for LR model
if "coef_" in dir(model):
    key = "final_coef_{reg}".format(**locals())
    data_fname = "{key}.tab".format(**locals())
    with open(data_fname, "w") as coef_out:
        coef_out.write("feature_name\tlabel\tcoef\n")
        for category_index, category in enumerate(model.classes_):
            label = category
            feature_name = "intercept"
            coef = model.intercept_[category_index]
            coef_out.write("{feature_name}\t{label}\t{coef}\n".format(**locals()))
            for feature_index, feature_name in enumerate(feature_names):
                coef = model.coef_[category_index, feature_index]
                coef_out.write("{feature_name}\t{label}\t{coef}\n".format(**locals()))

    script_fname = workdir / "{key}.R".format(**locals())
    script = \
"""
require(Cairo)
require(ggplot2)
require(RColorBrewer)

data <- read.delim("{data_fname}", header=TRUE)

data$coef_disc <- cut(data$coef, breaks=c(-Inf, -1.5, -0.2, 0.2, 1.5, Inf))
#levels(data$coef_disc) <- rev(levels(data$coef_disc))
palette="RdBu"
colors <- brewer.pal(palette, n=length(levels(data$coef_disc)))
color_mapping <- levels(data$coef_disc)
colors <- rev(colors)

p <- ggplot(data, aes(x=label, y=feature_name, fill=coef_disc)) +
  geom_tile() +
  theme_bw() +
  scale_fill_manual(values=colors, name="Model\ncoefficient", breaks=color_mapping, drop=FALSE) +
  scale_x_discrete(name="Label") +
  scale_y_discrete(name="") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave("{key}.pdf", p, width=8, height=5, units="in")
""".format(**locals())

    with open(script_fname, "w") as f:
        f.write(script)

    cmd = ["Rscript", script_fname]
    logger.info(" ".join(cmd))
    subprocess.check_call(cmd)


