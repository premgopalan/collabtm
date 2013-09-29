#ifndef COLLABTM_HH
#define COLLABTM_HH

#include "env.hh"
#include "ratings.hh"
#include "gammapoisson.hh"

class CollabTM {
public:
  CollabTM(Env &env, Ratings &ratings);
  ~CollabTM() { fclose(_af); }
  
  void batch_infer();

private:
  void initialize();
  void approx_log_likelihood();
  void get_phi(GPBase<Matrix> &a, uint32_t ai, 
	       GPBase<Matrix> &b, uint32_t bi, 
	       Array &phi);
  void get_tau(GPBase<Matrix> &a, GPBase<Matrix> &b,
	       uint32_t nd, D3 &tau);
  void update_all_rates();
  void swap_all();
  void compute_all_expectations();

  Env &_env;
  Ratings &_ratings;

  uint32_t _nusers;
  uint32_t _ndocs;
  uint32_t _nvocab;
  
  uint32_t _k;
  uint32_t _iter;

  GPMatrix _theta;
  GPMatrixGR _beta;
  GPMatrixGR _x;
  GPMatrixGR _epsilon;
  GPArray _a;
  D3 _tau;

  uint32_t _start_time;
  gsl_rng *_r;
  FILE *_af;
};

#endif
