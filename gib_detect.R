
# Load gib_detect_train.R
source("gib_detect_train.R")

# Load model data
model_data <- readRDS("gib_model.rds")

# Main loop
while (TRUE) {
  # Read input
  l <- readline(prompt = "")
  
  # Calculate transition probability and compare with threshold
  model_mat <- model_data$mat
  threshold <- model_data$thresh
  print(avg_transition_prob(l, model_mat) > threshold)
}
