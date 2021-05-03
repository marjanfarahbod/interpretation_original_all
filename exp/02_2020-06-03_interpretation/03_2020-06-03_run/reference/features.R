
require(Cairo)
require(reshape2)
require(ggplot2)
require(RColorBrewer)

data_fname <- "classifier_data.tab"
data <- read.delim(data_fname, header=TRUE, check.names=FALSE)

for (col in names(data)[3:length(names(data))]) {
    data[,col] <- data[,col] - mean(data[,col])
    data[,col] <- data[,col] / sd(data[,col])
}

label_mapping_fname <- "label_mappings.txt"
label_mapping <- read.delim(label_mapping_fname, header=TRUE)
for (i in 1:nrow(label_mapping)) {
    concatenation_key <- label_mapping[i, "concatenation_key"]
    orig_label <- label_mapping[i, "orig_label"]
    bio_label <- label_mapping[i, "new_label"]
    mask <- (as.character(data$concatenation_key) == concatenation_key) & (as.character(data$orig_label) == orig_label)
    data[mask, "bio_label"] <- bio_label
}

for (i in 1:nrow(data)) {
    data[i, "label_str"] <- paste(data[i, "bio_label"], data[i, "orig_label"], data[i, "concatenation_key"], sep="-")
}

melt_data <- melt(data, id.vars=c("concatenation_key", "orig_label", "bio_label", "label_str"))

melt_data$variable <- factor(melt_data$variable, levels = sort(levels(melt_data$variable)))
melt_data$value_disc <- cut(melt_data$value, breaks=c(-Inf, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, Inf))
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
ggsave(paste("reference/features_ecdf.pdf", sep=""), p, width=10, height=6, units="in")

for (bio_label in levels(melt_data$bio_label)) {
    bio_label_data <-  melt_data[melt_data$bio_label == bio_label,]
    if (nrow(bio_label_data) >= 30) { # XXX
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
        ggsave(paste("reference/features_bylabel_", bio_label, ".pdf", sep=""), p, width=9, height=plot_height, units="in")
    }
}

for (concatenation_key in levels(melt_data$concatenation_key)) {
    if (nrow(melt_data[melt_data$concatenation_key == concatenation_key,]) > 0) {
        plot_height <- 3 + 0.0125*nrow(melt_data[melt_data$concatenation_key == concatenation_key,])
        p <- ggplot(melt_data[melt_data$concatenation_key == concatenation_key,]) +
          aes(y=variable, x=label_str, fill=value_disc) +
          geom_tile() +
          scale_fill_manual(values=colors, breaks=color_mapping, drop=FALSE) +
          theme_bw() +
          theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
        ggsave(paste("reference/features_byann_", concatenation_key, ".pdf", sep=""), p, width=9, height=plot_height, units="in")
    }
}

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
ggsave(paste("reference/features_all.pdf", sep=""), p, width=9, height=plot_height, units="in", limitsize=FALSE)

    