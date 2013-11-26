#!/usr/bin/perl

my $bin = "/disk/scratch1/prem/collabtm-repo/src/collabtm";
my $loc = "/disk/scratch1/prem/collabtm-repo/analysis/mendeley";

my @cmds = ("$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -doc-only",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -doc-only",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -ratings-only -fixeda",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -ratings-only -fixeda",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -fixeda",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -fixeda",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100",
	    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10");

my @cmds2 = ("$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -vb",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -vb -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -vbinit 10 -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vbinit 10 -fixeda");

my @cmds3 = (	    
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -fixeda",
#    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -fixeda",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100",
#    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb -fixeda",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vbinit 10 -fixeda",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -doc-only",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -ratings-only -fixeda");

my @cmds4 = (
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -doc-only",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -ratings-only -fixeda",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -doc-only",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -ratings-only -fixeda"
    );

my @cmds5 = (
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 500",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 250",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 500 -fixeda",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 250 -fixeda",
    );

my @cmds6 = ("$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -doc-only",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -doc-only",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -ratings-only -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -ratings-only -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10 -fixeda",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 10",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb",
	     "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb -fixeda");

my @cmds2 = (
    #"$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -lda -fixeda -label defprior",
    #"$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -lda-init -label defprior",
    #"$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -lda-init -fixeda -label defprior");
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 300",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 300 -fixeda",
    "$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 300 -fixeda -vb");
    #"$bin -dir $loc -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -vb");



foreach my $cmd (@cmds2) {
    print "CMD = $cmd\n";
    system("$cmd 2>&1 > /dev/null &");
}
	    
