
require(Cairo)
require(ggplot2)
require(RColorBrewer)

data <- read.delim("xval/confusion.tab", header=TRUE)
data$true_label <- as.factor(data$true_label)
data$predicted_label <- as.factor(data$predicted_label)
data$true_label <- ordered(data$true_label, c('Quiescent', 'ConstitutiveHet', 'FacultativeHet', 'Transcribed', 'Promoter', 'Enhancer', 'RegPermissive', 'Bivalent', 'Unclassified'))
data$predicted_label <- ordered(data$predicted_label, c('Quiescent', 'ConstitutiveHet', 'FacultativeHet', 'Transcribed', 'Promoter', 'Enhancer', 'RegPermissive', 'Bivalent', 'Unclassified'))
data$reg <- as.factor(data$reg)

for (true_label in levels(data$true_label)) {
    data[data$true_label == true_label, "assignment_frequency"] <- data[data$true_label == true_label, "assignments"] / (sum(data[data$true_label == true_label, "assignments"]) / length(levels(data$reg)))
}

data$assignment_frequency_disc <- cut(data$assignment_frequency, breaks=c(-Inf, 0, 0.1, 0.4, 1))
palette="Reds"
colors <- brewer.pal(palette, n=length(levels(data$assignment_frequency_disc)))
color_mapping <- levels(data$assignment_frequency_disc)

for (reg in levels(data$reg)) {
    p <- ggplot(data[data$reg == reg, ], aes(x=true_label, y=predicted_label, fill=assignment_frequency_disc, label=assignments)) +
      geom_tile() +
      geom_text(size=4) +
      scale_fill_manual(values=colors, name="Frequency", breaks=color_mapping, drop=FALSE) +
      scale_x_discrete(name="True label") +
      scale_y_discrete(name="Predicted label") +
      coord_fixed() +
      theme_classic() +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
    ggsave(paste("xval/confusion_", reg, ".pdf", sep=""), p, width=5, height=4, units="in")
}
