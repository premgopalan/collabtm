#ifndef GAPREC_HH
#define GAPREC_HH

#include "env.hh"
#include "ratings.hh"

class GAPRec {
public:
  GAPRec(Env &env, Ratings &ratings);
  ~GAPRec();

  void batch_infer();
  void infer();

private:
  void initialize();
  void set_to_prior_users(Matrix &a, Matrix &b);
  void set_to_prior_movies(Matrix &a, Matrix &b);

  void set_etheta_sum();
  void set_ebeta_sum();

  void set_gamma_exp1(const Matrix &a, const Matrix &b, Matrix &v);
  void set_gamma_exp2(const Matrix &a, const Matrix &b, Matrix &v);
  
  void approx_log_likelihood();
  
  void update_global_state();
  double pair_likelihood(uint32_t p, uint32_t q, yval_t y) const;
  void test_likelihood();
  void validation_likelihood();
  uint32_t factorial(uint32_t n) const;  

  void init_heldout();
  void set_test_sample(int s);
  void set_validation_sample(int s);
  void get_random_rating1(Rating &r) const;
  void get_random_rating2(Rating &r) const;
  uint32_t duration() const;
  bool rating_ok(const Rating &e) const;
  bool is_test_rating(const Rating &e) const;
  string ratingslist_s(SampleMap &m);

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
  Matrix _bcurr;
  Matrix _ccurr;
  Matrix _dcurr;
  Matrix _anext;
  Matrix _bnext;
  Matrix _cnext;
  Matrix _dnext;

  Matrix _Elogtheta;
  Matrix _Etheta;
  Matrix _Elogbeta;
  Matrix _Ebeta;

  gsl_rng *_r;
  FILE *_hf;
  FILE *_vf;
  FILE *_af;

  SampleMap _test_map;
  RatingList _test_ratings;
  SampleMap _validation_map;
  RatingList _validation_ratings;

  Array _etheta_sum;
  Array _ebeta_sum;
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
  const SampleMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return false;
  const SampleMap::const_iterator w = _validation_map.find(r);
  if (w != _validation_map.end())
    return false;
  return true;
}

inline bool
GAPRec::is_test_rating(const Rating &r) const
{
  assert (r.first  < _n && r.second < _m);
  const SampleMap::const_iterator u = _test_map.find(r);
  if (u != _test_map.end())
    return true;
  return false;
}

#endif
