#include "env.hh"
#include "collabtm.hh"
#include "ratings.hh"

#include <stdlib.h>
#include <string>
#include <sstream>
#include <signal.h>

string Env::prefix = "";
Logger::Level Env::level = Logger::DEBUG;
FILE *Env::_plogf = NULL;
void usage();
void test();

void postprocess(Env &);

Env *env_global = NULL;
volatile sig_atomic_t sig_handler_active = 0;

void
term_handler(int sig)
{
  if (env_global) {
    printf("Got signal. Saving model state.\n");
    fflush(stdout);
    env_global->save_state_now = 1;
  } else {
    signal(sig, SIG_DFL);
    raise(sig);
  }
}

int
main(int argc, char **argv)
{
  signal(SIGTERM, term_handler);
  if (argc <= 1)
    exit(0);

  string fname;
  uint32_t n = 0, m = 0;
  uint32_t k = 0;
  string ground_truth_fname;
  uint32_t rfreq = 10;
  string label;
  bool logl = false;
  uint32_t max_iterations = 1000;
  bool nmi = false;
  bool strid = false;
  double rand_seed = 0;

  bool test = false;
  bool batch = true;
  bool online = false;
  bool gen_heldout = false;

  bool model_load = false;
  string model_location = "";

  bool hol_load = false;
  string hol_location = "";
  
  bool pred_accuracy = false;
  bool gt_accuracy = false;
  bool p = false;
  double a = 0.3, b = 0.3, c = 0.3, d = 0.3;
  Env::Dataset dataset = Env::MENDELEY;
  bool binary_data = false;
  bool explore = false;

  bool gen_ranking_for_users = false;
  bool fixeda = false;
  bool vbinit = false;
  uint32_t vbinit_iterations = 10;
  bool vb = false;
  bool use_docs = true;
  bool use_ratings = true;
  bool perturb_only_beta_shape = false;

  uint32_t nusers, ndocs, nvocab;
  bool lda = false;
  bool lda_init = false;
  bool ppc = false;
  bool seq_init = false;
  bool seq_init_samples = false;
  bool fixed_doc_param = false;
  bool phased = false;

  uint32_t i = 0;
  while (i <= argc - 1) {
    if (strcmp(argv[i], "-dir") == 0) {
      fname = string(argv[++i]);
      fprintf(stdout, "+ dir = %s\n", fname.c_str());
    } else if (strcmp(argv[i], "-nusers") == 0) {
      nusers = atoi(argv[++i]);
      fprintf(stdout, "+ nusers = %d\n", nusers);
    } else if (strcmp(argv[i], "-ndocs") == 0) {
      ndocs = atoi(argv[++i]);
      fprintf(stdout, "+ ndocs = %d\n", ndocs);
    } else if (strcmp(argv[i], "-nvocab") == 0) {
      nvocab = atoi(argv[++i]);
      fprintf(stdout, "+ nvocab = %d\n", nvocab);
    } else if (strcmp(argv[i], "-k") == 0) {
      k = atoi(argv[++i]);
      fprintf(stdout, "+ k = %d\n", k);
    } else if (strcmp(argv[i], "-nmi") == 0) {
      ground_truth_fname = string(argv[++i]);
      fprintf(stdout, "+ ground truth fname = %s\n",
	      ground_truth_fname.c_str());
      nmi = true;
    } else if (strcmp(argv[i], "-rfreq") == 0) {
      rfreq = atoi(argv[++i]);
      fprintf(stdout, "+ rfreq = %d\n", rfreq);
    } else if (strcmp(argv[i], "-strid") == 0) {
      strid = true;
      fprintf(stdout, "+ strid mode\n");
    } else if (strcmp(argv[i], "-label") == 0) {
      label = string(argv[++i]);
    } else if (strcmp(argv[i], "-logl") == 0) {
      logl = true;
      fprintf(stdout, "+ logl mode\n");
    } else if (strcmp(argv[i], "-max-iterations") == 0) {
      max_iterations = atoi(argv[++i]);
      fprintf(stdout, "+ max iterations %d\n", max_iterations);
    } else if (strcmp(argv[i], "-seed") == 0) {
      rand_seed = atof(argv[++i]);
      fprintf(stdout, "+ random seed set to %.5f\n", rand_seed);
    } else if (strcmp(argv[i], "-load") == 0) {
      model_load = true;
      model_location = string(argv[++i]);
      fprintf(stdout, "+ loading theta from %s\n", model_location.c_str());
    } else if (strcmp(argv[i], "-test") == 0) {
      test = true;
      fprintf(stdout, "+ test mode\n");
    } else if (strcmp(argv[i], "-batch") == 0) {
      batch = true;
      fprintf(stdout, "+ batch inference\n");
    } else if (strcmp(argv[i], "-online") == 0) {
      batch = false;
      fprintf(stdout, "+ online inference\n");
    } else if (strcmp(argv[i], "-gen-heldout") == 0) {
      gen_heldout = true;
      fprintf(stdout, "+ generate held-out files from dataset\n");
    } else if (strcmp(argv[i], "-pred-accuracy") == 0) {
      pred_accuracy = true;
      fprintf(stdout, "+ compute predictive accuracy\n");
    } else if (strcmp(argv[i], "-gt-accuracy") == 0) {
      gt_accuracy = true;
      fprintf(stdout, "+ compute  accuracy to ground truth\n");
    } else if (strcmp(argv[i], "-netflix") == 0) {
      dataset = Env::NETFLIX;
    } else if (strcmp(argv[i], "-mendeley") == 0) {
      dataset = Env::MENDELEY;
    } else if (strcmp(argv[i], "-movielens") == 0) {
      dataset = Env::MOVIELENS;
    } else if (strcmp(argv[i], "-echonest") == 0) {
      dataset = Env::ECHONEST;
    } else if (strcmp(argv[i], "-binary-data") == 0) {
      binary_data = true;
    } else if (strcmp(argv[i], "-explore") == 0) {
      explore = true;
    } else if (strcmp(argv[i], "-gen-ranking") == 0) {
      gen_ranking_for_users = true;
    } else if (strcmp(argv[i], "-fixeda") == 0) {
      fixeda = true;
    } else if (strcmp(argv[i], "-vbinit") == 0) {
      vbinit = true;
      vbinit_iterations = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-vb") == 0) {
      vb = true;
    } else if (strcmp(argv[i], "-doc-only") == 0) {
      use_docs = true;
      use_ratings = false;
    } else if (strcmp(argv[i], "-ratings-only") == 0) {
      use_docs = false;
      use_ratings = true;
    } else if (strcmp(argv[i], "-init0") == 0) {
      perturb_only_beta_shape = true;
    } else if (strcmp(argv[i], "-lda") == 0) {
      lda = true;
    } else if (strcmp(argv[i], "-lda-init") == 0) {
      lda_init = true;
    } else if (strcmp(argv[i], "-ppc") == 0) {
      ppc = true;
    } else if (strcmp(argv[i], "-seq-init") == 0) {
      seq_init = true;
    } else if (strcmp(argv[i], "-seq-init-samples") == 0) {
      seq_init_samples = true;
    } else if (strcmp(argv[i], "-fixed-doc-param") == 0) {
      fixed_doc_param = true;
    } else if (strcmp(argv[i], "-phased") == 0) {
      phased = true;
    } else if (i > 0) {
      fprintf(stdout,  "error: unknown option %s\n", argv[i]);
      assert(0);
    } 
    ++i;
  };

  Env env(ndocs, nvocab, nusers, k, 
	  fname, nmi, ground_truth_fname, rfreq, 
	  strid, label, logl, rand_seed, max_iterations, 
	  model_load, model_location, 
	  gen_heldout, dataset,
	  batch, binary_data, vb, explore, 
	  fixeda, vbinit, vbinit_iterations,
	  use_docs, use_ratings, perturb_only_beta_shape,
	  lda, lda_init, ppc, seq_init, seq_init_samples,
	  fixed_doc_param, phased);
  
  env_global = &env;
  if (p) {
    postprocess(env);
    exit(0);
  }
  
  Ratings ratings(env);
  if (ratings.read(fname.c_str()) < 0) {
    fprintf(stderr, "error reading dataset from dir %s; quitting\n", 
	    fname.c_str());
    return -1;
  }
  
  if (!ppc) {
    CollabTM collabtm(env, ratings);
    collabtm.batch_infer();
  } else {
    CollabTM collabtm(env, ratings);
    collabtm.ppc();
  }
  exit(0);
}

