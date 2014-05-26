require(ggplot2)
require(data.table)
require(plyr)
require(scales)

theme_set(theme_bw())

########################################
# function to compute precision at various number of recommendations by user
########################################
compute.precision.by.user <- function(recommendations, users, num.recs=seq(10,100,10)) {
  precision.by.user <- ddply(recommendations, "user", function (df) {
    df <- df[order(-df$predicted), ]
    df$hits <- cumsum(df$actual)
    data.frame(num.recs=num.recs, precision=df[num.recs, "hits"] / num.recs)
  })
  precision.by.user <- merge(precision.by.user, users, by="user", all.x=T)
}


########################################
# function to compute number of hits (# recs in the test set) by user
########################################
compute.hits.by.user <- function(recommendations, num.recs=seq(10,100,10)) {
  hits.by.user <- ddply(recommendations, "user", function (df) {
    df <- df[order(-df$predicted), ]
    df$hits <- cumsum(df$actual)
    data.frame(num.recs=num.recs, hits=df[num.recs, "hits"])
  })
}


########################################
# function to compute coverage at various number of recommendations by item
########################################
compute.coverage.by.item <- function(recommendations, items, num.recs=seq(10,100,10)) {
  # get item rank for each hit by user
  hits <- ddply(recommendations, "user", function(df) {
    df <- df[order(-df$predicted), ]
    df$rank <- 1:nrow(df)
    df <- subset(df, actual == 1)
  })

  # compute how many users have each item in their top-n recommendations
  coverage.by.item <- ddply(hits, "item", function(df) {
    df <- adply(num.recs, 1, function(n) {
      data.frame(num.users=sum(df$rank <= n))
    })
    df$X1 <- num.recs
    colnames(df) <- c("num.recs","num.users.covered")
    df
  })

  # compute fraction of users covered by top-n recommendations for each item
  test.items <- ddply(recommendations, "item", summarize, total.users=length(user))
  coverage.by.item <- merge(coverage.by.item, test.items, by="item", all.x=T)
  coverage.by.item <- transform(coverage.by.item, coverage=num.users.covered/total.users)

  # join on item popularity in training set
  coverage.by.item <- merge(coverage.by.item, items, by="item", all.x=T)
}


########################################
# evaluate methods across different ranks
########################################
args <- commandArgs(trailingOnly=T)

if (length(args) > 0) {
  ranks <- eval(parse(text=sprintf('c(%s)', args[1])))
} else {
  ranks <- c(100)
}

# compute precision and coverage, writing to separate data frames
for (dataset in c("echonest", "nyt", "netflix", "mendeley", "netflix45")) {
  # read user activity and item popularity
  users.file <- sprintf('../data/%s/users.tsv', dataset)
  users <- read.delim(users.file, sep='\t', header=F, col.names=c('user','activity'))
  items.file <- sprintf('../data/%s/items.tsv', dataset)
  items <- read.delim(items.file, sep='\t', header=F, col.names=c('item','popularity'))
  test.users.file <- sprintf('../data/%s/test_user_degree.tsv', dataset)
  test.users <- read.delim(test.users.file, sep='\t', header=F, col.names=c('user','num.test.items'))

  # notes:
  # netflix45 is implicit, where only ratings >= 4 are treated as (binary) observations
  # netflix is explicit, where all rating values are modeled
  #  the method here is "mf" instead of "mfpop", as no negative sampling is used,
  #  so output/netflix/mf is symlinked to output/netflix/mfpop
  # in both cases the test set involves prediction of ratings >= 4 (items users "like")
  for (method in c("bpf.hier", "bpf", "lda", "nmf", "mfpop")) {
    for (K in ranks) {
      ranking.file <- sprintf('../output/%s/%s/ranking.tsv', dataset, method)
      prec.file <- sprintf('../output/%s/%s/precision.txt', dataset, method)
      recall.file <- sprintf('../output/%s/%s/recall.txt', dataset, method)
      coverage.file <- sprintf('../output/%s/%s/coverage.txt', dataset, method)

      if (T) {
      #if (file.exists(ranking.file) && !file.exists(recall.file)) {
        print(ranking.file)
        predictions <- read.delim(ranking.file, sep='\t', header=F, col.names=c('user','item','predicted','actual'))

        hits.by.user <- compute.hits.by.user(predictions)

        precision.by.user <- merge(hits.by.user, users, by="user", all.x=T)
        #precision.by.user <- transform(precision.by.user, precision=hits/num.recs)
        precision.by.user <- merge(precision.by.user, test.users, by="user", all.x=T)
	precision.by.user$precision <- apply(precision.by.user[,c('num.test.items','num.recs')],1,min)
        precision.by.user <- transform(precision.by.user, precision=hits/precision)
        precision.by.user$method <- toupper(method)
        precision.by.user$K <- K
        precision.by.user$dataset <- dataset

        write.table(precision.by.user, file=prec.file, row.names=F)

        recall.by.user <- merge(hits.by.user, test.users, by="user", all.x=T)
        recall.by.user <- transform(recall.by.user, recall=hits/num.test.items)
        recall.by.user <- merge(recall.by.user, users, by="user", all.x=T)
        recall.by.user$method <- toupper(method)
        recall.by.user$K <- K
        recall.by.user$dataset <- dataset

        write.table(recall.by.user, file=recall.file, row.names=F)

        #coverage.by.item <- compute.coverage.by.item(predictions, items)
        #coverage.by.item$method <- toupper(method)
        #coverage.by.item$K <- K
        #coverage.by.item$dataset <- dataset
        #write.table(coverage.by.item, file=coverage.file, row.names=F)
	}
    }
  }
}

