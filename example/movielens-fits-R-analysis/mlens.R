library(Matrix)
printf <- function(...)print(sprintf(...))
  
show_top_movies <- function(u)
{
  movies.of.user <- merge(users, titles, by="itemid")
  movies.of.user <- subset(movies.of.user, movies.of.user$userid == u)
  a <- movies.of.user[order(-movies.of.user$rating)[1:20],]
  print(na.omit(a[c("title", "type")]), row.names=FALSE)
}

show_top_user_factors <- function(u)
{
  k <- 100
  df <- data.frame(cbind(c(1:k), as.matrix(t(subset(theta, theta$userid == u)))[3:(k+2)]))
  colnames(df) <- c("fac", "theta")
  betam <- as.matrix(beta[, -c(1,2)])
  facs <- as.matrix(df$fac[order(df$theta, decreasing=T)[1:100]])
  n <- 0
  for (c in facs[1:100]) {
      # skip unused factor
      q <- length(which(as.logical(beta[[c]])))
      if (q <= 1) {
        next;
      }
      top_movies_by_factor(c)
      n <- n + 1
      if (n > 2) {
        break;
      }
  }
}

show_related_movies <- function(moviename)
{
  mname <- paste(moviename, "(,|\\s+).*", sep="")
  itemid <- grep(mname, titles$title, perl=TRUE)
  if (length(itemid) > 1) {
    itemid <- itemid[1]
  }
  type <- as.character(titles[itemid,"type"])
  itemid <- as.matrix(titles[itemid,]["itemid"])
  itemid <- as.integer(itemid[1][1])
  betam <- as.matrix(beta[, -c(1,2)])
  movseq <- which(beta$itemid == itemid)
  printf("item = %d, title = %s, type = %s", 
         itemid, moviename, type)
  df <- data.frame(cbind(c(1:100), betam[movseq,]))
  colnames(df) <- c("seq", "beta")
  top.mov.seq <- df$seq[order(df$beta, decreasing=T)] 
  for (c in top.mov.seq[1:3]) {
    top_movies_by_factor(c)
  }
  barplot(betam[movseq,])
}

top_movies_by_factor <- function(x)
{ 
  printf("FACTOR %d", x)
  factorname <- sprintf("K%d",x)
  a <- merge(beta, titles, by="itemid")
  a <- a[order(-a[[factorname]])[1:10],]
  print(na.omit(a[c("title", "type")]), row.names=FALSE)
}