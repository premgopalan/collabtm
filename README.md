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

The binary 'gaprec' will be installed in /usr/local/bin unless a
different prefix is provided to configure. (See INSTALL.)

GAPREC: Gamma Poisson factorization based recommendation tool
--------------------------------------------------------------

**gaprec** [OPTIONS]

    -dir <string>    path to dataset directory with 3 files: train.tsv, test.tsv, validation.tsv (for examples, see example/movielens-1m)
 
    -m <int>	     number of items
    -n <int>	     number of users
    -k <int>	     number of factors
   
    -rfreq <int>     assess convergence and compute other stats; <int> number of iterations; default: 10

    -a
    -b		     set hyperparameters
    -c		     default: a = b = c = d = 0.3
    -d

    -bias	     use user and item bias terms

    -binary-data     treat observed data as binary; if rating > 0 then rating is treated as 1

    -gen-ranking     generate ranking file to use in precision computation; see example		  


Example
--------

../src/gaprec -dir ../example/movielens -n 6040 -m 3900  -k 100 -rfreq 10

This will write output in n6040-m3900-k100-batch:

* param.txt: shows hyperparameter and other settings
* precision.txt: computed mean precision at 10 and 100 on the test.tsv
* validation.txt: the log likelihood on the validation.tsv ratings
* theta.txt, beta.txt: the approx. expected posterior Poisson parameters
* a,b,c,d.txt: the approx. expected posterior Gamma parameters
* Ei, Eu.txt: the approx. expected bias parameters, if any (default none)
* infer.log: monitor inference progress

To generate the ranking file (ranking.tsv) for precision computation,
run the following:

cd n6040-m3900-k100-batch;
../../src/gaprec -dir ../../example/movielens -n 6040 -m 3900  -k 100 -rfreq 10 -gen-ranking

This will rank all y == 0 in training and the test.tsv pairs in
decreasing order of their scores, along with true ratings from
test.tsv.

The output is now in n6040-m3900-k100-batch/ranking.tsv.

Bias terms
----------

To model user activity and item popularity, bias parameters are
additionally inferred for each user and item. The new \theta and \beta
are as follows:
    		  
\theta_u' = [\theta_u, 1, g_u]
\beta_i' =  [\beta_i, h_i, 1]   


