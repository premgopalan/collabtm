

# global objects
nusers <- 10000
ntitles <- 261248
ndocs <- 10000
nvocab <- 10000

theta <- data.frame(read.table("theta.tsv"))
colnames(theta) <- c("seq", "docid")

beta <- data.frame(read.table("beta.tsv"))
colnames(beta) <- c("seq", "wordid")

epsilon <- data.frame(read.table("epsilon.tsv"))
colnames(epsilon) <- c("seq", "docid")

x <- data.frame(read.table("x.tsv"))
colnames(x) <- c("seq", "userid")

titles <- data.frame(readLines("titles.dat"))
titles <- data.frame(cbind(c(0:(ntitles-1)), titles))
colnames(titles) <- c("docid", "title")

words <- data.frame(read.table("vocab.dat"))
words <- data.frame(cbind(c(0:(nvocab-1)), words))
colnames(words) <- c("wordid", "word")

beta.with.words <- merge(beta, words, by="wordid")
colnames(beta.with.words) <- c("seq", "wordid", c(1:10), "word")

epsilon.with.titles <- merge(epsilon, titles, by="docid")
colnames(epsilon.with.titles) <- c("seq", "docid", c(1:10), "title")

theta.with.titles <- merge(theta, titles, by="docid")
colnames(theta.with.titles) <- c("seq", "docid", c(1:10), "title")

# functions
theta_top_titles <- function(factor)
{
  c <- sprintf("%d", factor) 
  s <- theta.with.titles[order(-theta.with.titles[[c]])[1:30],]
  print(na.omit(s[c("title", factor)]), row.names=FALSE)
}

epsilon_top_titles <- function(factor)
{
  c <- sprintf("%d", factor)
  t <- epsilon.with.titles[order(-epsilon.with.titles[[c]])[1:30],]
  print(na.omit(t[c("title", factor)]), row.names=FALSE)
}

topics <- function(factor)
{
  c <- sprintf("%d", factor)
  t <- beta.with.words[order(-beta.with.words[[c]])[1:30],]
  print(na.omit(t[c("word", factor)]), row.names=FALSE)
}

