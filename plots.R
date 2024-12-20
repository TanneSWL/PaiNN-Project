rm(list = ls())

library(tidyverse)
library(latex2exp)
library(ggrepel)
library(patchwork)

#-------------------------------------------------------------------------------
# Number if layers vs. performance

layers <- c(1, 3, 5, 7, 9, 11, 13, 15)
MAE <- list()

# Extract the results for test MAE for each run. 
for(i in 1:length(layers)){
  MAE[i] <- suppressWarnings(readLines(paste("results/output_", layers[i], "L/test_MAE.csv", sep = "")))
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
data_pred <- read.table("results/output_9L/predictions_vs_true_labels_pandas.csv",
                        sep = ",",
                        header = TRUE)

# Compute MAE.
data_pred <- tibble(data_pred) |>
  rename(True = "True.Labels") |>
  mutate(MAE = True - Predictions)

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
  ylab("Error [eV]") +
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
         Atoms = str_count(Smiles, "[A-Z]") - str_count(Smiles, "H"),
         Nitrogen_frac = Nitrogen/Atoms,
         absMAE = abs(MAE))

data_pred_out <- data_pred |>
  filter(abs(MAE) > 0.25) |>
  arrange(MAE) |>
  rownames_to_column(var = "Number") 
  
# Plot results. 
lookForPatter <- function(feature, xlabel){
  ggplot(data = data_pred,
         mapping = aes(x = {{feature}},
                       y = absMAE)) +
    geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
    geom_point(color = "#CC6600",
               size = 2,
               alpha = 0.5) +
    xlab( xlabel) + 
    ylab("Absolute Error [eV]") +
    theme(axis.title.x = element_text(margin = margin(t = 20), size = 14, face = "bold"),
          axis.title.y = element_text(margin = margin(r = 20), size = 14, face = "bold"),
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10))
}


p1 <- lookForPatter(Positive, "Number of Positive Charges")

p2 <- lookForPatter(Negative, "Number of Negative Charges") +
  scale_x_continuous(limits = c(0, 2), breaks = seq(0, 2, by = 1))

p3 <- lookForPatter(Total, "Total Number of Charges")

(p1 / p2 / p3) + plot_annotation(tag_levels = 'A')

ggsave("charges.png",
       width = 8,
       height = 12)

p4 <- lookForPatter(Nitrogen, "Number of Nitrogen Atoms") + 
  scale_x_continuous(limits = c(0, 7), breaks = seq(0, 7, by = 1))

p5 <- lookForPatter(Atoms, "Number of Heavy Atoms") +
  scale_x_continuous(limits = c(0, 9), breaks = seq(0, 9, by = 1))

p6 <- lookForPatter(Nitrogen_frac, "Number of Nitrogen / Number of Atoms")

(p4 / p5 / p6) + plot_annotation(tag_levels = 'A')

ggsave("nitrogen.png",
       width = 8,
       height = 12)

data_pred |>
  filter(Atoms == 9) |>
  count()
