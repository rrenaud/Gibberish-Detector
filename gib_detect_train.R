# Define accepted characters
accepted_chars <- c(letters, " ")

# Create a dictionary of character positions
pos <- setNames(1:length(accepted_chars), accepted_chars)

# Normalize function
normalize <- function(line) {
  gsub(paste0("[^", paste(accepted_chars, collapse = ""), "]"), "", tolower(line))
}

# Ngram function
ngram <- function(n=2, line) {
  filtered <- normalize(line)
  sapply(1:(nchar(filtered) - n + 1), function(start) {
    substr(filtered, start, start + n - 1)
  })
}

# Train function
train <- function() {
  k <- length(accepted_chars)
  # Initialize counts
  counts <- matrix(10, nrow = k, ncol = k)
  
  # Count transitions from big text file
  big_text <- readLines("big.txt")
  for (line in big_text) {
    #line <- big_text[1]
    grams <- ngram(2, line)
    for (gram in grams) {
      #gram <- grams[2]
      a <- substr(gram, 1, 1)
      b <- substr(gram, 2, 2)
      counts[pos[a], pos[b]] <- counts[pos[a], pos[b]] + 1
    }
  }
  
  # Normalize counts
  counts <- log(counts / rowSums(counts))
  
  # Calculate average transition probabilities for good and bad phrases
  good_text <- readLines("good.txt")
  #l <- good_text[1]
  good_probs <- sapply(good_text, function(l) {
    avg_transition_prob(l, counts)
  })
  bad_text <- readLines("bad.txt")
  bad_probs <- sapply(bad_text, function(l) {
    avg_transition_prob(l, counts)
  })
  
  # Assert that good phrases have higher probabilities than bad phrases
  stopifnot(min(good_probs) > max(bad_probs))
  
  # Pick a threshold halfway between the worst good and best bad inputs
  thresh <- (min(good_probs) + max(bad_probs)) / 2
  
  # Save model
  saveRDS(list(mat = counts, thresh = thresh), file = "gib_model.rds")
}

# Average transition probability function
avg_transition_prob <- function(l, log_prob_mat) {
  grams <- ngram(2, l)
  log_prob <- sapply(grams, function(gram){
    a <- substr(gram, 1, 1)
    b <- substr(gram, 2, 2)
    log_prob_mat[pos[a],pos[b]]
  })
  log_prob <- sum(log_prob)
  transition_ct <- length(grams)
  exp(log_prob / max(transition_ct, 1))
}

# Call train function
train()
