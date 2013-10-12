#!/usr/bin/perl

my $bin = "/disk/scratch1/prem/collabtm/src/collabtm";
my $loc = "/disk/scratch1/prem/collabtm/analysis/mendeley";

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


foreach my $cmd (@cmds) {
    system("$cmd 2>&1 > /dev/null &");
}
	    
