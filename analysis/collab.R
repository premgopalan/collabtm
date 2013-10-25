# set the directory with the model fit 

#dir <- "nusers80278-ndocs261248-nvocab10000-k100-batch-RATINGS-ONLY"
dir <- "nusers80278-ndocs261248-nvocab10000-k100-batch-DOC-ONLY"  

# SET ratings or docs (not both, not yet)
ratings <- 0
docs <- 1

# set the number of factors
K <- 100

# global objects
nusers <- 80278
ntitles <- 261248
ndocs <- 261248
nvocab <- 10000


if (docs) {
   ttitles <- data.frame(readLines("mendeley/titles.dat"))
   ttitles <- data.frame(cbind(c(0:(ntitles-1)), ttitles))
   colnames(ttitles) <- c("docid", "title")

  thetaf <- sprintf ("%s/theta.tsv", dir)
  theta <- data.frame(read.table(thetaf))
  colnames(theta) <- c("seq", "docid")

  betaf <- sprintf ("%s/beta.tsv", dir)
  beta <- data.frame(read.table(betaf))
  colnames(beta) <- c("seq", "wordid")

  words <- data.frame(read.table("mendeley/vocab.dat"))
  words <- data.frame(cbind(c(0:(nvocab-1)), words))
  colnames(words) <- c("wordid", "word")

  beta.with.words <- merge(beta, words, by="wordid")
  colnames(beta.with.words) <- c("wordid", "seq", c(1:K), "word")

  theta.with.titles <- merge(theta, ttitles, by="docid")
  colnames(theta.with.titles) <- c("docid", "seq", c(1:K), "title")

  } 

if (ratings) {
   etitles <- data.frame(readLines("mendeley/titles.dat"))
   etitles <- data.frame(cbind(c(1:ntitles), etitles))
   colnames(etitles) <- c("docid", "title")

  epsilonf <- sprintf ("%s/epsilon.tsv", dir)
  epsilon <- data.frame(read.table(epsilonf))
  colnames(epsilon) <- c("seq", "docid")

  #xf <- sprintf ("%s/x.tsv", dir)
  #x <- data.frame(read.table(xf))
  #colnames(x) <- c("seq", "userid")
  
  epsilon.with.titles <- merge(epsilon, etitles, by="docid")
  colnames(epsilon.with.titles) <- c("docid", "seq", c(1:K), "title")
}

# functions
theta_top_titles <- function(factor)
{
  c <- sprintf("%d", factor)
  s <- theta.with.titles[order(-theta.with.titles[[c]])[1:100],]
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
  t <- beta.with.words[order(-beta.with.words[[c]])[1:100],]
  print(na.omit(t[c("word", factor)]), row.names=FALSE)
}

