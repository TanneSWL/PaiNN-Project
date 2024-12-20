rm(list = ls())

library(tidyverse)
library(latex2exp)
library(ggrepel)

#-------------------------------------------------------------------------------
# Number if layers vs. performance

layers <- c(1, 3, 5, 7, 9, 11, 13, 15)
MAE <- list()

# Extract the results for test MAE for each run. 
for(i in 1:length(layers)){
  MAE[i] <- suppressWarnings(readLines(paste("output_", layers[i], "L/test_MAE.csv", sep = "")))
}

MAE <- as.numeric(MAE)

# Convert to tibble.
data_performance <- tibble(layers = layers,
                           MAE = MAE)

# Plot results
ggplot(data = data_performance,
       mapping = aes(x = layers,
                     y = MAE)) + 
  geom_path(color = "#030F4F",
            size = 1) + 
  geom_point(color = "#030F4F",
             fill = "#CC6600",
             size = 4,
             shape = 21) +
  xlab("Number of Message Passing Layers") + 
  ylab("MAE [meV]") +
  scale_x_continuous(breaks = layers) +
  theme(axis.title.x = element_text(margin = margin(t = 20), size = 14, face = "bold"),
        axis.title.y = element_text(margin = margin(r = 20), size = 14, face = "bold"),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

# Save results.
ggsave("MAE_layers.png",
       width = 8,
       height = 4)

# ------------------------------------------------------------------------------
# Worse predictions

# Get predictions for 9 layers.
data_pred <- read.table("output_9L/predictions_vs_true_labels_pandas.csv",
                        sep = ",",
                        header = TRUE)

# Compute MAE.
data_pred <- tibble(data_pred) |>
  rename(True = "True.Labels") |>
  mutate(MAE= True - Predictions)

# Extract the worse predictions in a seperate set.
data_pred_out <- data_pred |>
  filter(abs(MAE) > 0.25) |>
  arrange(MAE) |>
  rownames_to_column(var = "Number") 

# Plot results. 
ggplot(data = data_pred,
       mapping = aes(x = True,
                     y = MAE)) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  geom_point(color = "#CC6600",
             size = 2,
             alpha = 0.5) +
  xlab("True Internal Energy (0 K) [eV]") + 
  ylab("MAE [eV]") +
  theme(axis.title.x = element_text(margin = margin(t = 20), size = 14, face = "bold"),
        axis.title.y = element_text(margin = margin(r = 20), size = 14, face = "bold"),
        axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) + 
  geom_point(data = data_pred_out,
             mapping = aes(x = True,
                           y = MAE),
             color = "#030F4F",
             size = 4,
             shape = 21) + 
  geom_text_repel(data = data_pred_out,
                  mapping = aes(x = True,
                                y = MAE,
                                label = Number))
# Save results.
ggsave("MAE_true.png",
       width = 8,
       height = 4)

# ------------------------------------------------------------------------------
# Structure characteristics vs. performance

# We use the SMILE strings to look for other patterns.

# Add new features for the test observations. 
data_pred <- data_pred |>
  mutate(Positive = str_count(Smiles, "\\+"),
         Negative = str_count(Smiles, "-"),
         Total = Positive + Negative,
         Nitrogen = str_count(Smiles, "N"),
         Atoms = str_count(Smiles, "[A-Z]"),
         Nitrogen_frac = Nitrogen/Atoms)

data_pred_out <- data_pred |>
  filter(abs(MAE) > 0.25) |>
  arrange(MAE) |>
  rownames_to_column(var = "Number") 
  
# Plot results. 
lookForPatter <- function(feature, xlabel){
  ggplot(data = data_pred,
         mapping = aes(x = {{feature}},
                       y = MAE)) +
    geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
    geom_point(color = "#CC6600",
               size = 2,
               alpha = 0.5) +
    xlab( xlabel) + 
    ylab("MAE [eV]") +
    theme(axis.title.x = element_text(margin = margin(t = 20), size = 14, face = "bold"),
          axis.title.y = element_text(margin = margin(r = 20), size = 14, face = "bold"),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10)) + 
    geom_point(data = data_pred_out,
               mapping = aes(x = {{feature}},
                             y = MAE),
               color = "#030F4F",
               size = 4,
               shape = 21) + 
    geom_text_repel(data = data_pred_out,
                    mapping = aes(x = {{feature}},
                                  y = MAE,
                                  label = Number))
}


lookForPatter(Positive, "Number of Positive Charges")
ggsave("positive.png",
       width = 8,
       height = 4)


lookForPatter(Negative, "Number of Negative Charges")
ggsave("negative.png",
       width = 8,
       height = 4)

lookForPatter(Total, "Total Number of Charges")
ggsave("total.png",
       width = 8,
       height = 4)

lookForPatter(Nitrogen, "Number of Nitrogen Atoms")
ggsave("nitrogen.png",
       width = 8,
       height = 4)

lookForPatter(Atoms, "Number of Atoms")
ggsave("atoms.png",
       width = 8,
       height = 4)

lookForPatter(Nitrogen_frac, "Number of Nitrogen / Number of Atoms")
ggsave("nitrogen_frac.png",
       width = 8,
       height = 4)
