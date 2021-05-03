
require(Cairo)
require(ggplot2)
#require(RColorBrewer)

data <- read.delim("./xval/scores.tab", header=TRUE)

p <- ggplot(data) +
  aes_string(x="reg", y="accuracy") +
  stat_summary(fun.data="mean_cl_boot") +
  scale_x_log10(name="Regularization") +
  scale_y_continuous(name="Accuracy") +
  theme_bw()
ggsave("xval/scores.pdf", p, width=4, height=3, units="in")
