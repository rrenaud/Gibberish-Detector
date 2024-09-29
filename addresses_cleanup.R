library(data.table)
addresses <- fread("addresses.csv")
#cleanup addresses#####
#big_text_raw <- head(addresses)
big_text_raw <- addresses[1:100000,]
accepted_chars <- c(letters, " ")
normalize <- function(line) {
  gsub(paste0("[^", paste(accepted_chars, collapse = ""), "]"), "", tolower(line))
}
big_text <- mapply(normalize, big_text_raw$building_name)
big_text <- data.table(words = big_text)
big_text_cleaned <- strsplit(big_text$words," ")
big_text_cleaned <- unlist(big_text_cleaned)
big_text_cleaned <- data.table(big_text_cleaned)
big_text_cleaned$len <- sapply(big_text_cleaned$big_text_cleaned,nchar)
big_text_cleaned <- big_text_cleaned[len>=5]
fwrite(big_text_cleaned,"addresses_cleaned.csv")
