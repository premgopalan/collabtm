#ifndef GAPREC_HH
#define GAPREC_HH

#include "env.hh"
#include "ratings.hh"

class GAPRec {
public:
  GAPRec(Env &env, Ratings &ratings);
  ~GAPRec();

  void batch_infer();
  void batch_infer_bias();
  void infer();
  void analyze();
  void analyze_factors();
  void gen_ranking_for_users();


private:
  void initialize();
  void set_to_prior_users(Matrix &a, Array &b);
  void set_to_prior_movies(Matrix &a, Array &b);
  
  void initialize_bias();
  void set_to_prior_biases();
  void update_global_state_bias();
  void set_gamma_exp_bias(const Array &u, double v, Array &w1, Array &w2);
  double link_prob_bias(uint32_t user, uint32_t movie) const;
  double pair_likelihood_bias(uint32_t p, uint32_t q, yval_t y) const;


  void save_model();
  void write_movie_list(string label, uint32_t u, const vector<uint32_t> &movies);
  void write_movie_list(string label, const vector<uint32_t> &movies);

  void save_user_state(string s, const Matrix &mat);
  void save_item_state(string s, const Matrix &mat);
  void save_state(string s, const Array &mat);
  void do_on_stop();

  void load_file(string s, Matrix &mat);
  void load_beta_and_theta();
  void load_file(string s, Array &mat);


  void set_etheta_sum();
  void set_ebeta_sum();
  void set_etheta_sum2();
  void set_ebeta_sum2();
  void load_validation_and_test_sets();

  void set_gamma_exp(const Matrix &a, const Array &b, Matrix &v1, Matrix &v2);
  void set_gamma_exp_init(const Matrix &a, Matrix &v1, Matrix &v2, double v);
  void set_gamma_exp1(const Matrix &a, const Array &b, Matrix &v);
  void set_gamma_exp2(const Matrix &a, const Array &b, Matrix &v);
  void set_gamma_exp1_idx(uint32_t u, 
			  const Matrix &a, const Matrix &b, Matrix &v);
  void set_gamma_exp2_idx(uint32_t u, 
			  const Matrix &a, const Matrix &b, Matrix &v);

  void set_gamma_exp2_array(uint32_t u, 
			    const Matrix &a, const Array &b, Matrix &v);
  void set_gamma_exp1_array(uint32_t u, 
			    const Matrix &a, const Array &b, Matrix &v);

  void recompute_etheta_sum(const UserMap &sampled_users, 
			    const Array &etheta_sum_old);

  void compute_etheta_sum(const UserMap &sampled_users);
  void compute_ebeta_sum(const MovieMap &sampled_movies);
  void adjust_etheta_sum(const UserMap &sampled_users);
  void adjust_ebeta_sum(const MovieMap &sampled_movies);


  void optimize_user_rate_parameters(uint32_t n);
  void optimize_user_shape_parameters_helper(uint32_t n, 
					     Matrix &cphi);
  void optimize_user_shape_parameters(uint32_t n);
  void optimize_user_shape_parameters2(uint32_t n);

  
  void approx_log_likelihood();
  void auc(bool bias = false);
  double link_prob(uint32_t user, uint32_t movie) const;
  
  void update_global_state();
  double pair_likelihood(uint32_t p, uint32_t q, yval_t y) const;
  double pair_likelihood_binary(uint32_t p, uint32_t q, yval_t y) const;
  void test_likelihood();
  void validation_likelihood(bool bias = false);
  uint32_t factorial(uint32_t n) const;  

  void init_heldout();
  void set_test_sample(int s);
  void set_training_sample();
  void set_validation_sample(int s);
  void get_random_rating1(Rating &r) const;
  void get_random_rating2(Rating &r) const;
  void get_random_rating3(Rating &r) const;
  uint32_t duration() const;
  bool rating_ok(const Rating &e) const;
  bool is_test_rating(const Rating &e) const;
  void write_sample(FILE *, SampleMap &m);
  void write_sample(FILE *, CountMap &m);
  void write_ebeta(uint32_t);
  void write_etheta(uint32_t);

  Env &_env;
  Ratings &_ratings;

  uint32_t _n;
  uint32_t _m;
  uint32_t _k;
  uint32_t _iter;

  double _a;
  double _b;
  double _c;
  double _d;

  time_t _start_time;

  Matrix _acurr;
  Array _bcurr;
  Matrix _ccurr;
  Array _dcurr;
  Matrix _anext;
  Array _bnext;
  Matrix _cnext;
  Array _dnext;
  Array _phi;

  Array _ucurr;
  Array _icurr;
  Array _unext;
  Array _inext;
  Array _Elogu;
  Array _Eu;
  Array _Elogi;
  Array _Ei;

  Matrix _Elogtheta;
  Matrix _Etheta;
  Matrix _Elogbeta;
  Matrix _Ebeta;

  gsl_rng *_r;
  FILE *_hf;
  FILE *_vf;
  FILE *_af;
  FILE *_pf;

  CountMap _test_map;
  RatingVec _test_ratings;
  CountMap _validation_map;
  RatingVec _validation_ratings;
  UserMap _sampled_users;
  UserMap _sampled_movies;

  Array _etheta_sum;
  Array _ebeta_sum;
  Array _etheta_sum_old;
  Array _ebeta_sum_old;
  double _tau0;
  double _kappa;
  Array _rho;
  
  uint32_t _nh;
  double _prev_h;
  bool _save_ranking_file;
  uArray _itemc;
  bool _use_rate_as_score;
  uint32_t _topN_by_user;
};

inline uint32_t
GAPRec::duration() const
{
  time_t t = time(0);
  return t - _start_time;
}

inline bool
GAPRec::rating_ok(const Rating &r) const
{
  assert (r.first  < _n && r.second < _m);
  const CountMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return false;
  const CountMap::const_iterator w = _validation_map.find(r);
  if (w != _validation_map.end())
    return false;
  return true;
}

inline bool
GAPRec::is_test_rating(const Rating &r) const
{
  assert (r.first  < _n && r.second < _m);
  const CountMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return true;
  return false;
}



#endif
