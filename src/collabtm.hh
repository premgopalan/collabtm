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
  void ppc();

private:
  void initialize();
  void initialize_perturb_betas();
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

  void seq_init();
  void seq_init_helper();
  void write_coldstart_docs(FILE *f, MovieMap &mp);

  void load_validation_and_test_sets();
  void save_user_state(string s, const Matrix &mat);
  void save_item_state(string s, const Matrix &mat);
  void save_state(string s, const Array &mat);
  void compute_likelihood(bool validation);
  double per_rating_likelihood(uint32_t user, uint32_t doc, yval_t y) const;
  double coldstart_ratings_likelihood(uint32_t user, uint32_t doc) const;
  uint32_t duration() const;
  bool rating_ok(const Rating &r) const;
  uint32_t factorial(uint32_t n)  const;
  double log_factorial(uint32_t n)  const;

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
  FILE *_vf;
  FILE *_tf;
  double _prev_h;
  uint32_t _nh;

  CountMap _validation_map;  
  CountMap _test_map;
  MovieMap _cold_start_docs;
};

inline uint32_t
CollabTM::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline bool
CollabTM::rating_ok(const Rating &r) const
{
  assert (r.first  < _nusers && r.second < _ndocs);
  const CountMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return false;
  const CountMap::const_iterator w = _validation_map.find(r);
  if (w != _validation_map.end())
    return false;
  return true;
}


#endif
