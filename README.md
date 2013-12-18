Installation
------------

Required libraries: gsl, gslblas, pthread

On Linux/Unix run

 ./configure
 make; make install

On Mac OS, the location of the required gsl, gslblas and pthread
libraries may need to be specified:

 ./configure LDFLAGS="-L/opt/local/lib" CPPFLAGS="-I/opt/local/include"
 make; make install

The binary 'collabtm' will be installed in /usr/local/bin unless a
different prefix is provided to configure. (See INSTALL.)

COLLABTM: Nonnegative Collaborative Topic Modeling tool
--------------------------------------------------------

**collabtm** [OPTIONS]

    -dir <string>            path to dataset directory with files described under INPUT below
 
    -mdocs <int>	     number of documents

    -nuser <int>	     number of users

    -nvocab <int>	     size of vocabulary
    	    
    -k <int>                 latent dimensionality

    -fixeda                  fix the document length correction factor ('a') to 1

    -binary-data             treat observed ratings data as binary; if rating > 0 then rating is treated as 1

    -doc-only                use document data only

    -ratings-only            use ratings data only

    -lda-init                use LDA based initialization (see below)

    -seq-init -doc-only	     use sequential initialization for document only fits

INPUT 
-----

We need the same files used by the basic factorization model.

train.tsv, test.tsv, validation.tsv, test_users.tsv

The new files additionally needed are mult.dat and vocab.dat.  (They are really text files.) This is the "document" portion of the data. Each line of mult.dat is a document and has the following format:

     <number of words> <word-id0:count0> <word-id1:count1>....

Each line of vocab.dat is a word. Note that both the word index and the document index starts at 0. So a word-id in vocab.dat can be 0 and the document id "rated" in train.tsv can be 0.

EXAMPLE
-------

Run two versions -- with the correction scalar 'a' inferred and one with 'a' fixed at a 1.  One of these fits might be better than the other. 

/disk/scratch1/prem/collabtm/src/collabtm -dir /disk/scratch1/prem/collabtm/analysis/mendeley -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100

/disk/scratch1/prem/collabtm/src/collabtm -dir /disk/scratch1/prem/collabtm/analysis/mendeley -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -fixeda 


LDA BASED INITIALIZATION
------------------------

1. Run Chong's gibbs sampler to obtain LDA fits on the word frequencies.

2. Create a directory "lda-fits" within the "dataset directory" above and put
two files in it: the topics beta-lda-k<K>.tsv and the memberships
theta-lda-k<K>.tsv.  If K=100, these files will be named beta-lda-k100.tsv and
theta-lda-k100.tsv, respectively.

3. Run collabtm inference with the -lda-init option as follows (the -fixeda option fixes 'a' at 1):

/disk/scratch1/prem/collabtm/src/collabtm -dir /disk/scratch1/prem/collabtm/analysis/mendeley -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -lda-init

/disk/scratch1/prem/collabtm/src/collabtm -dir /disk/scratch1/prem/collabtm/analysis/mendeley -nusers 80278 -ndocs 261248 -nvocab 10000 -k 100 -lda-init -fixeda


CHONG's GIBBS SAMPLER
---------------------

See the package under "lda" directory.
