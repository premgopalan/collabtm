require(ggplot2)
require(data.table)
require(plyr)
require(scales)

theme_set(theme_bw())

methods <- c("HPF", "BPF", "LDA", "MF", "NMF")
datasets <- c("mendeley"="Mendeley","nyt"="New York Times","echonest"="Echo Nest","netflix45"="Netflix (implicit)","netflix"="Netflix (explicit)")

########################################
# plot simple descriptives
########################################

#
# users
#

# read user activity by dataset
users <- adply(names(datasets), 1, function(dataset) {
  tsv <- sprintf('../data/%s/users.tsv', dataset)
  users <- read.table(tsv, header=F, col.names=c('user', 'activity'))
})
users$X1 <- names(datasets)[users$X1]
names(users)[1] <- "dataset"

# clean up dataset labels and remove netflix implicit
users <- transform(users, dataset=revalue(dataset, datasets))
users <- transform(users, dataset=factor(as.character(dataset), datasets))
users <- subset(users, dataset != "Netflix (implicit)")
users <- transform(users, dataset=revalue(dataset, c("Netflix (explicit)"="Netflix")))

# plot distribution of user activity by dataset
plot.data <- ddply(users, c("dataset","activity"), summarize, num.users=length(user))
p <- ggplot(data=plot.data, aes(x=activity, y=num.users))
p <- p + geom_point()
p <- p + scale_x_log10(labels=comma, breaks=10^(0:4)) + scale_y_log10(labels=comma, breaks=10^(0:6))
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('User activity') + ylab('Number of users')
ggsave(p, filename='../../KDD-paper/figures/user_activity.pdf', width=10, height=2.5)
p

# plot cdf of user activity by dataset
plot.data <- ddply(users, c("dataset","activity"), summarize, num.users=length(user))
plot.data <- ddply(plot.data, "dataset", transform, frac.users=rev(cumsum(rev(num.users)))/sum(num.users))
p <- ggplot(data=plot.data, aes(x=activity, y=frac.users))
p <- p + geom_line()
p <- p + scale_x_log10(labels=comma, breaks=10^(0:3))
p <- p + scale_y_continuous(labels=percent)
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('Number of user views') + ylab('Fraction of users')
ggsave(p, filename='../../KDD-paper/figures/user_activity_cdf.pdf', width=10, height=2.5)
p

###
### broken due to change in format for sim_byuser.tsv in netflix ###
###
if (F) {
# read bpf simulated user activity by dataset
sim.users <- adply(names(datasets), 1, function(dataset) {
  tsv <- sprintf('../marginals/%s/sim_byusers.tsv', dataset)
  print(tsv)
  read.table(tsv, header=F, col.names=c("rep", "seq", "user", "activity", "total.ratings"))
})
sim.users$X1 <- names(datasets)[sim.users$X1]
names(sim.users)[1] <- "dataset"
sim.users <- transform(sim.users, dataset=revalue(dataset, datasets))
sim.users <- subset(sim.users, dataset != "Netflix (implicit)")
sim.users <- transform(sim.users, dataset=revalue(dataset, c("Netflix (explicit)"="Netflix")))


# plot distribution of user activity for bpf simulated activity
plot.data <- subset(sim.users, rep==0)
plot.data <- rbind(cbind(users[, c("dataset","user","activity")], variable=rep('Empirical', nrow(users))),
                   cbind(plot.data[, c("dataset","user","activity")], variable=rep('Simulated', nrow(plot.data))))
plot.data <- ddply(plot.data, c("variable", "dataset","activity"), summarize, num.users=length(user))
plot.data <- ddply(plot.data, c("variable","dataset"), transform, frac.users=rev(cumsum(rev(num.users)))/sum(num.users))
p <- ggplot(data=plot.data, aes(x=activity, y=frac.users))
p <- p + geom_line(aes(linetype=variable))
p <- p + scale_x_log10(labels=comma, breaks=10^(0:3))
p <- p + scale_y_continuous(labels=percent)
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('Number of user views') + ylab('Fraction of users')
p <- p + theme(legend.title=element_blank(), legend.position=c(0.94,0.8), legend.background=element_blank())
ggsave(p, filename='../../KDD-paper/figures/user_activity_sim.pdf', width=10, height=2.5)
p

# plot cdf of user activity for bpf simulated activity
 plot.data <- ddply(plot.data, c("variable", "dataset","activity"), summarize, num.users=length(user))
p <- ggplot(data=plot.data, aes(x=activity, y=num.users))
p <- p + geom_point(aes(shape=variable, color=variable))
p <- p + scale_x_log10(labels=comma, breaks=10^(0:4)) + scale_y_log10(labels=comma, breaks=10^(0:6))
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('User activity') + ylab('Number of users')
p <- p + theme(legend.title=element_blank(), legend.position=c(0.94,0.8), legend.background=element_blank())
ggsave(p, filename='../../KDD-paper/figures/user_activity_sim_cdf.pdf', width=10, height=2.5)
p
}
###
### end broken ###
###

#
# netflix implict: empirical, mf, and hpf marginals
#

netflix.sim.users <- adply(c("netflix45"), 1, function(dataset) {
  tsv <- sprintf('../data/%s/users.tsv', dataset)
  users <- read.table(tsv, header=F, col.names=c('id', 'tratings'))
  users$method <- "Empirical"

  tsv <- sprintf('../marginals/%s/gauss_sim_byusers2.tsv', dataset)
  mf.sim.users <- read.table(tsv, header=T, col.names=c("rep","seq","id","count","tratings","tratings_mean"))
  mf.sim.users$method <- "MF"

  tsv <- sprintf('../marginals/%s/sim_byusers.tsv', dataset)
  hpf.sim.users <- read.table(tsv, header=T)
  hpf.sim.users$method <- "HPF"

  sim.users <- rbind(users,
                     mf.sim.users[, c("id", "tratings", "method")],
                     hpf.sim.users[, c("id", "tratings", "method")])
})
netflix.sim.users$X1 <- "Netflix"
names(netflix.sim.users) <- c("dataset", "user", "activity", "method")
netflix.sim.users <- transform(netflix.sim.users, dataset=revalue(dataset, datasets))


# plot cdf of user activity for bpf and mf simulated activity
plot.data <- ddply(netflix.sim.users, c("method", "dataset","activity"), summarize, num.users=length(user))
plot.data <- ddply(plot.data, c("method","dataset"), transform, frac.users=rev(cumsum(rev(num.users)))/sum(num.users))
p <- ggplot(data=plot.data, aes(x=activity, y=frac.users))
p <- p + geom_line(aes(color=method, linetype=method))
p <- p + scale_x_log10(labels=comma, breaks=10^(0:3))
p <- p + scale_y_continuous(labels=percent)
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('Number of user views') + ylab('Fraction of users')
p <- p + theme(legend.title=element_blank(), legend.position=c(0.2,0.3), legend.background=element_blank())
ggsave(p, filename='../../KDD-paper/figures/user_activity_sim_cdf_netflix.pdf', width=4, height=3)
p


# plot distribution of user activity for bpf and mf simulated activity
plot.data <- transform(netflix.sim.users, bin=10^(round(log10(activity)*15)/15))
plot.data <- ddply(plot.data, c("method", "dataset","bin"), summarize, num.users=length(user))
p <- ggplot(data=subset(plot.data, method != "Empirical"), aes(x=bin, y=num.users))
p <- p + geom_point(data=subset(plot.data, method == "Empirical"), size=1, shape=0)
p <- p + geom_line(aes(linetype=method, color=method))
p <- p + scale_x_log10(labels=comma, breaks=10^(0:4)) + scale_y_log10(labels=comma, breaks=10^(0:6))
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('User activity') + ylab('Number of users')
p <- p + theme(legend.title=element_blank(), legend.position=c(0.2,0.3), legend.background=element_blank())
ggsave(p, filename='../../KDD-paper/figures/user_activity_sim_netflix.pdf', width=4, height=3)
p


#
# items
#

# read item popularity by dataset
items <- adply(names(datasets), 1, function(dataset) {
  tsv <- sprintf('../data/%s/items.tsv', dataset)
  items <- read.table(tsv, header=F, col.names=c('item', 'popularity'))
})
items$X1 <- names(datasets)[items$X1]
names(items)[1] <- "dataset"
items <- transform(items, dataset=revalue(dataset, datasets))
items <- transform(items, dataset=factor(as.character(dataset), datasets))

# clean up dataset labels and remove netflix implicit
items <- transform(items, dataset=revalue(dataset, datasets))
items <- transform(items, dataset=factor(as.character(dataset), datasets))
items <- subset(items, dataset != "Netflix (implicit)")
items <- transform(items, dataset=revalue(dataset, c("Netflix (explicit)"="Netflix")))

# plot distribution of item activity by dataset
plot.data <- ddply(items, c("dataset","popularity"), summarize, num.items=length(item))
p <- ggplot(data=plot.data, aes(x=popularity, y=num.items))
p <- p + geom_point()
p <- p + scale_x_log10(labels=comma, breaks=10^(0:4)) + scale_y_log10(labels=comma, breaks=10^(0:6))
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('Item popularity') + ylab('Number of items')
ggsave(p, filename='../../KDD-paper/figures/item_popularity.pdf', width=10, height=2.5)
p

# plot cdf of item popularity by dataset
plot.data <- ddply(items, c("dataset","popularity"), summarize, num.items=length(item))
plot.data <- ddply(plot.data, "dataset", transform, frac.items=rev(cumsum(rev(num.items)))/sum(num.items))
p <- ggplot(data=plot.data, aes(x=popularity, y=frac.items))
p <- p + geom_line()
p <- p + scale_x_log10(labels=comma, breaks=10^(0:4))
p <- p + scale_y_continuous(labels=percent)
p <- p + facet_wrap(~ dataset, nrow=1)
p <- p + xlab('Number of item views') + ylab('Fraction of items')
ggsave(p, filename='../../KDD-paper/figures/item_popularity_cdf.pdf', width=10, height=2.5)
p



########################################
# read precision and coverage data frames
########################################

precision.by.user <- data.frame()
recall.by.user <- data.frame()
coverage.by.item <- data.frame()
for (dataset in names(datasets)) {
  print(dataset)

  # notes:
  # netflix45 is implicit, where only ratings >= 4 are treated as (binary) observations
  # netflix is explicit, where all rating values are modeled
  #  the method here is "mf" instead of "mfpop", as no negative sampling is used,
  #  so output/netflix/mf is symlinked to output/netflix/mfpop
  # in both cases the test set involves prediction of ratings >= 4 (items users "like")
  for (method in c("bpf.hier", "bpf", "lda", "nmf", "mfpop")) {
      tsv <- sprintf('../output/%s/%s/precision.txt', dataset, method)
      if (file.exists(tsv)) {
        print(tsv)
        precision.by.user <- rbind(precision.by.user,
                                   read.table(tsv, header=T))
      } else {
        print(sprintf("%s not found", tsv))
      }

      tsv <- sprintf('../output/%s/%s/recall.txt', dataset, method)
      if (file.exists(tsv)) {
        print(tsv)
        recall.by.user <- rbind(recall.by.user,
                                read.table(tsv, header=T))
      } else {
        print(sprintf("%s not found", tsv))
      }
    }
}


#######################################
# preprocessing
########################################

# remove users with missing activity from the training file
precision.by.user <- subset(precision.by.user, !is.na(activity) & !is.na(num.test.items))
recall.by.user <- subset(recall.by.user, !is.na(activity) & !is.na(num.test.items))

# keep only mfpop, renaming to mf
# clean up dataset names
precision.by.user <- subset(precision.by.user, method != "MFUNIF")
precision.by.user <- transform(precision.by.user,
                               method=revalue(method, c("BPF.HIER"="HPF", "MFPOP"="MF")),
                               dataset=revalue(dataset, datasets))
recall.by.user <- subset(recall.by.user, method != "MFUNIF")
recall.by.user <- transform(recall.by.user,
                            method=revalue(method, c("BPF.HIER"="HPF", "MFPOP"="MF")),
                            dataset=revalue(dataset, datasets))

# set order of methods and datasets for all plots
precision.by.user <- transform(precision.by.user,
                               dataset=factor(as.character(dataset), datasets),
                               method=factor(as.character(method), methods))
recall.by.user <- transform(recall.by.user,
                            dataset=factor(as.character(dataset), datasets),
                            method=factor(as.character(method), methods))


# make all evaluations for rank 100 and 20 recommendations
N <- 20
rank <- 100

########################################
# precision/recall at N
########################################

# plot mean precision at N recs for methods and datasets
plot.data <- subset(precision.by.user, num.recs==N & K==rank)
plot.data <- ddply(plot.data, c("dataset","method","K","num.recs"), summarize, mean.precision=mean(precision))
p <- ggplot(plot.data, aes(x=dataset, y=mean.precision))
p <- p + geom_point(aes(color=method), size=1, show_guide=F)
p <- p + geom_hline(aes(yintercept=mean.precision, colour=method, linetype=method), size=1, show_guide=T)
p <- p + facet_wrap(~ dataset, nrow=1, scale="free")
p <- p + xlab("") + ylab('Mean normalized precision')
p <- p + scale_y_continuous(labels=percent)
p <- p + theme(legend.title=element_blank())
p <- p + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_precision_at_%d.pdf', N), width=10, height=2.5)
p


# plot mean recall at N recs for methods and datasets
plot.data <- subset(recall.by.user, num.recs==N & K==rank)
plot.data <- ddply(plot.data, c("dataset","method","K","num.recs"), summarize, mean.recall=mean(recall))
p <- ggplot(plot.data, aes(x=dataset, y=mean.recall))
p <- p + geom_point(aes(color=method), size=1, show_guide=F)
p <- p + geom_hline(aes(yintercept=mean.recall, colour=method, linetype=method), size=1, show_guide=T)
p <- p + facet_wrap(~ dataset, nrow=1, scale="free")
p <- p + xlab("") + ylab('Mean recall')
p <- p + scale_y_continuous(labels=percent)
p <- p + theme(legend.title=element_blank())
p <- p + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_recall_at_%d.pdf', N), width=10, height=2.5)
p


########################################
# precision/recall by number of recs
########################################

# plot mean precision by number of recs for methods and datasets
plot.data <- subset(precision.by.user, K==rank)
plot.data <- ddply(plot.data, c("dataset","method","K","num.recs"), summarize, mean.precision=mean(precision))
p <- ggplot(plot.data, aes(x=num.recs, y=mean.precision))
p <- p + geom_line(aes(linetype=as.factor(method), colour=as.factor(method)))
p <- p + xlab('Number of recommendations') + ylab('Mean normalized precision')
p <- p + scale_x_continuous(breaks=c(10,50,100)) + scale_y_continuous(labels=percent)
p <- p + theme(legend.title=element_blank())
p <- p + facet_wrap(~ dataset, nrow=1, scale="free_y")
ggsave(p, filename='../../KDD-paper/figures/mean_precision_by_num_recs.pdf', width=10, height=2.5)
p


# plot mean recall by number of recs for methods and datasets
plot.data <- subset(recall.by.user, K==rank)
plot.data <- ddply(plot.data, c("dataset","method","K","num.recs"), summarize, mean.recall=mean(recall))
p <- ggplot(plot.data, aes(x=num.recs, y=mean.recall))
p <- p + geom_line(aes(linetype=as.factor(method), colour=as.factor(method)))
p <- p + xlab('Number of recommendations') + ylab('Mean recall')
p <- p + scale_x_continuous(breaks=c(10,50,100)) + scale_y_continuous(labels=percent)
p <- p + theme(legend.title=element_blank())
p <- p + facet_wrap(~ dataset, nrow=1, scale="free_y")
ggsave(p, filename='../../KDD-paper/figures/mean_recall_by_num_recs.pdf', width=10, height=2.5)
p


########################################
# precision/recall by user activity
########################################

# plot mean precision by user activity percentile
percentiles <- seq(0.05,1,0.05)
plot.data <- subset(precision.by.user, num.recs==N)
plot.data <- ddply(plot.data, c("dataset","method"), function(df) {
  adply(percentiles, 1, function(p) {
    with(subset(df, activity <= quantile(activity, p)), mean(precision, na.rm=T))
  })
})
plot.data$X1 <- percentiles[plot.data$X1]
names(plot.data) <- c("dataset","method","percentile","mean.precision")
p <- ggplot(plot.data, aes(x=percentile, y=mean.precision))
p <- p + geom_line(aes(color=method, linetype=method))
p <- p + facet_wrap(~ dataset, nrow=1, scale="free_y")
p <- p + scale_x_continuous(labels=percent, breaks=c(0.1, 0.5, 0.9))
p <- p + scale_y_continuous(labels=percent)
p <- p + xlab('User percentile by activity') + ylab('Mean normalized precision')
p <- p + theme(legend.title=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_precision_at_%d_by_user_percentile.pdf', N), width=10, height=2.5)
p


# plot mean recall by user activity percentile
percentiles <- seq(0.05,1,0.05)
plot.data <- subset(recall.by.user, num.recs==N)
plot.data <- ddply(plot.data, c("dataset","method"), function(df) {
  adply(percentiles, 1, function(p) {
    with(subset(df, activity <= quantile(activity, p)), mean(recall, na.rm=T))
  })
})
plot.data$X1 <- percentiles[plot.data$X1]
names(plot.data) <- c("dataset","method","percentile","mean.recall")
p <- ggplot(plot.data, aes(x=percentile, y=mean.recall))
p <- p + geom_line(aes(color=method, linetype=method))
p <- p + facet_wrap(~ dataset, nrow=1, scale="free_y")
p <- p + scale_x_continuous(labels=percent, breaks=c(0.1, 0.5, 0.9))
p <- p + scale_y_continuous(labels=percent)
p <- p + xlab('User percentile by activity') + ylab('Mean recall')
p <- p + theme(legend.title=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_recall_at_%d_by_user_percentile.pdf', N), width=10, height=2.5)
p



########################################
# difference in precision/recall at N
########################################

diff.by.user <- data.table(precision.by.user)
diff.by.user <- merge(diff.by.user, subset(diff.by.user, method=="HPF"), by=c("dataset","num.recs","user"), suffixes=c('', '.hpf'))
diff.by.user <- diff.by.user[, diff.precision:=(hits - hits.hpf) / num.recs]
diff.by.user <- diff.by.user[, diff.recall:=(hits - hits.hpf) / num.test.items]


# plot mean precision at N recs for methods and datasets
plot.data <- subset(diff.by.user, num.recs==N & K==rank)
plot.data <- ddply(plot.data, c("dataset","method","K","num.recs"), summarize, mean.diff.precision=mean(diff.precision))
p <- ggplot(plot.data, aes(x=dataset, y=mean.diff.precision))
p <- p + geom_point(aes(color=method), size=1, show_guide=F)
p <- p + geom_hline(aes(yintercept=mean.diff.precision, colour=method, linetype=method), size=1, show_guide=T)
p <- p + facet_wrap(~ dataset, nrow=1, scale="free")
p <- p + xlab("") + ylab('Mean difference in precision')
#p <- p + scale_y_continuous(labels=percent)
p <- p + theme(legend.title=element_blank())
p <- p + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_diff_precision_at_%d.pdf', N), width=10, height=2.5)
p


# plot mean recall at N recs for methods and datasets
plot.data <- subset(diff.by.user, num.recs==N & K==rank)
plot.data <- ddply(plot.data, c("dataset","method","K","num.recs"), summarize, mean.diff.recall=mean(diff.recall))
p <- ggplot(plot.data, aes(x=dataset, y=mean.diff.recall))
p <- p + geom_point(aes(color=method), size=1, show_guide=F)
p <- p + geom_hline(aes(yintercept=mean.diff.recall, colour=method, linetype=method), size=1, show_guide=T)
p <- p + facet_wrap(~ dataset, nrow=1, scale="free")
p <- p + xlab("") + ylab('Mean difference in recall')
#p <- p + scale_y_continuous(labels=percent)
p <- p + theme(legend.title=element_blank())
p <- p + theme(axis.text.x=element_blank(), axis.ticks.x=element_blank(), axis.title.x=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_diff_recall_at_%d.pdf', N), width=10, height=2.5)
p



########################################
# difference in precision/recall by user activity
########################################


# plot mean difference in precision by user activity percentile
percentiles <- seq(0.05,1,0.05)
plot.data <- subset(diff.by.user, num.recs==N)
plot.data <- ddply(plot.data, c("dataset","method"), function(df) {
  adply(percentiles, 1, function(p) {
    with(subset(df, activity <= quantile(activity, p)), mean(diff.precision, na.rm=T))
  })
})
plot.data$X1 <- percentiles[plot.data$X1]
names(plot.data) <- c("dataset","method","percentile","mean.diff.precision")
p <- ggplot(plot.data, aes(x=percentile, y=mean.diff.precision))
p <- p + geom_line(aes(color=method, linetype=method))
p <- p + facet_wrap(~ dataset, nrow=1, scale="free_y")
p <- p + scale_x_continuous(labels=percent, breaks=c(0.1, 0.5, 0.9))
#p <- p + scale_y_continuous(labels=percent)
p <- p + xlab('User percentile by activity') + ylab('Mean difference in precision')
p <- p + theme(legend.title=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_diff_precision_at_%d_by_user_percentile.pdf', N), width=10, height=2.5)
p

# plot mean difference in recall by user activity percentile
percentiles <- seq(0.05,1,0.05)
plot.data <- subset(diff.by.user, num.recs==N)
plot.data <- ddply(plot.data, c("dataset","method"), function(df) {
  adply(percentiles, 1, function(p) {
    with(subset(df, activity <= quantile(activity, p)), mean(diff.recall, na.rm=T))
  })
})
plot.data$X1 <- percentiles[plot.data$X1]
names(plot.data) <- c("dataset","method","percentile","mean.diff.recall")
p <- ggplot(plot.data, aes(x=percentile, y=mean.diff.recall))
p <- p + geom_line(aes(color=method, linetype=method))
p <- p + facet_wrap(~ dataset, nrow=1, scale="free_y")
p <- p + scale_x_continuous(labels=percent, breaks=c(0.1, 0.5, 0.9))
#p <- p + scale_y_continuous(labels=percent)
p <- p + xlab('User percentile by activity') + ylab('Mean difference in recall')
p <- p + theme(legend.title=element_blank())
ggsave(p, filename=sprintf('../../KDD-paper/figures/mean_diff_recall_at_%d_by_user_percentile.pdf', N), width=10, height=2.5)
p

