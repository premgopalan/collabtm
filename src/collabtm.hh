#ifndef COLLABTM_HH
#define COLLABTM_HH

#include "env.hh"
#include "ratings.hh"
#include "gpbase.hh"

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
  void get_xi(uint32_t nu, uint32_t nd, 
	      Array &xi,
	      Array &xi_a, 
	      Array &x_b);
  void update_all_rates();
  void update_all_rates_in_seq();
  void swap_all();
  void compute_all_expectations();
  void save_model();
  
  void save_user_state(string s, const Matrix &mat);
  void save_item_state(string s, const Matrix &mat);
  void save_state(string s, const Array &mat);

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
  GPMatrix _epsilon;
  GPArray _a;

  uint32_t _start_time;
  gsl_rng *_r;
  FILE *_af;
};

#endif
