#ifndef COLLABTM_HH
#define COLLABTM_HH

#include "env.hh"
#include "ratings.hh"
#include "gpbase.hh"

class CollabTM {
public:
  CollabTM(Env &env, Ratings &ratings);
  ~CollabTM();
  
  void batch_infer();
  void online_infer(); 
  void write_mult_format();

  void gen_ranking_for_users(); 
  void ppc();

private:
  void initialize();
  void initialize_perturb_betas();
  void approx_log_likelihood();
  void precision(); 
  void coldstart_precision();

  void get_phi(GPBase<Matrix> &a, uint32_t ai, 
	       GPBase<Matrix> &b, uint32_t bi, 
	       Array &phi);
  void get_xi(uint32_t nu, uint32_t nd, 
	      Array &xi,
	      Array &xi_a, 
	      Array &x_b);
  void get_xi_decoupled(uint32_t nu, uint32_t nd, 
			Array &xi,
			Array &xi_a, 
			Array &x_b);
  void update_all_rates();
  void update_all_rates_in_seq();
  void stochastic_update_all_rates_in_seq(UserMap &sampled_words); 
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
  bool compute_likelihood(bool validation);
  double per_rating_likelihood(uint32_t user, uint32_t doc, yval_t y, 
			       bool coldstart=false) const;
  double per_rating_prediction(uint32_t user, uint32_t doc) const;

  uint32_t duration() const;
  bool is_training(const Rating &r) const;
  uint32_t factorial(uint32_t n)  const;
  double log_factorial(uint32_t n)  const;

  void coldstart_local_inference();
  void coldstart_rating_likelihood();
  double coldstart_per_rating_prediction(uint32_t user, uint32_t doc) const;

  bool is_validation(const Rating &r) const;
  void do_on_stop();

  Env &_env;
  Ratings &_ratings;

  uint32_t _nusers;
  uint32_t _ndocs;
  uint32_t _nvocab;
  
  uint32_t _k;
  uint32_t _iter;
  Array _rhou, _rhow;
  double _tau0, _kappa;
  uArray _userc, _wordc;

  GPMatrix _theta;
  GPMatrixGR _beta;
  GPMatrixGR _x;
  GPMatrixGR _xd;
  GPMatrix _epsilon;
  GPArray _a;
  GPMatrix *_cstheta; // coldstart docs
  
  uint32_t _start_time;
  gsl_rng *_r;
  FILE *_af;
  FILE *_vf;
  FILE *_tf;
  FILE *_pf;
  FILE *_cs_pf;
  FILE *_df;
  FILE *_cs_df;
  FILE *_cs_tf;
  FILE *_cs_tf2;
  double _prev_h;
  uint32_t _nh;
  bool _save_ranking_file;
  uint32_t _topN_by_user;

  CountMap _validation_map;  
  CountMap _test_map;
  CountMap _coldstart_test_map;
  MovieMap _cold_start_docs;
  vector<uint32_t> _cs_users;
  UserMap _sampled_users;
  UserMap _sampled_movies;
  uArray *_cs_test_users;

  // coldstart docs sequence ids
  uint32_t _ncsdoc_seq;
  IDMap _doc_to_cs_idmap;
};

inline
CollabTM::~CollabTM()
{
  fclose(_af);
  fclose(_vf);
  fclose(_tf);
  fclose(_pf);
  fclose(_df);
  fclose(_cs_tf);
  fclose(_cs_tf2);
  fclose(_cs_pf);
  fclose(_cs_df);
}

inline uint32_t
CollabTM::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline bool
CollabTM::is_training(const Rating &r) const
{
  assert (r.first  < _nusers && r.second < _ndocs);

  MovieMap::const_iterator mp = _cold_start_docs.find(r.second);
  if (mp != _cold_start_docs.end())
    return false;
  const CountMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return false;
  const CountMap::const_iterator w = _validation_map.find(r);
  if (w != _validation_map.end())
    return false;
  return true;
}

inline bool
CollabTM::is_validation(const Rating &r) const
{
  assert (r.first  < _nusers && r.second < _ndocs);
  CountMap::const_iterator itr = _validation_map.find(r);
  if (itr != _validation_map.end())
    return true;
  return false;
}


#endif
