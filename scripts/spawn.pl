#!/usr/bin/perl

my $bin = "/scratch/pgopalan/collabtm/src/collabtm";
my $datadir = "/scratch/pgopalan/collabtm/analysis/mendeley";

my @cmds = (
    "$bin -dir $datadir -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -fixeda -lda-init -vb",
    "$bin -dir $datadir -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -fixeda -lda-init -fixed-doc-param -vb",
    "$bin -dir $datadir -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -fixeda -lda-init -phased -vb",
    "$bin -dir $datadir -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -fixeda -ratings-only -vb",
    "$bin -dir $datadir -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -fixeda -lda-init -content-only -vb",
    "$bin -dir $datadir -nusers 80278 -ndocs 261248 -nvocab 10000 -k 200 -fixeda -lda-init -decoupled -vb",
    );

foreach my $cmd (@cmds) {
    print "CMD = $cmd\n";
    system("$cmd 2>&1 > /dev/null &");
}
