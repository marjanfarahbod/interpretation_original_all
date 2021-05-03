#!/usr/bin/env Rscript
## transcript produced by Segtools 1.2.4

## Experimental R transcript
## You may not be able to run the R code in this file exactly as written.

segtools.r.dirname <-
            system2("python",
            c("-c", "'import segtools; print segtools.get_r_dirname()'"),
            stdout = TRUE)

source(file.path(segtools.r.dirname, 'common.R'))
source(file.path(segtools.r.dirname, 'signal.R'))
source(file.path(segtools.r.dirname, 'track_statistics.R'))
save.track.stats('signal_distribution', 'signal_distribution', 'signal_distribution/signal_distribution.tab', gmtk = FALSE, as_regex = FALSE, mnemonic_file = '', clobber = FALSE, translation_file = '', track_order = list(), label_order = list())
