#include "gaprec.hh"

GAPRec::GAPRec(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _n(env.n), _m(env.m), _k(env.k),
    _iter(0),
    _a(0.3), _b(0.3),
    _c(0.3), _d(0.3),
    _start_time(time(0)),
    _acurr(_n,_k), _bcurr(_n,_k),
    _ccurr(_m,_k), _dcurr(_m,_k),
    _anext(_n,_k), _bnext(_n,_k),
    _cnext(_m,_k), _dnext(_m,_k),
    _Elogtheta(_n,_k), _Etheta(_n,_k),
    _Elogbeta(_m,_k), _Ebeta(_m,_k),
    _etheta_sum(_k),
    _ebeta_sum(_k)
{
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  Env::plog("infer n:", _n);

  _hf = fopen(Env::file_str("/heldout.txt").c_str(), "w");
  if (!_hf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _af = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_af)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  init_heldout();
}

GAPRec::~GAPRec()
{
  fclose(_hf);
  fclose(_vf);
  fclose(_af);
}

void
GAPRec::init_heldout()
{
  int s = _env.heldout_ratio * _ratings.nratings();
  set_test_sample(s);
  Env::plog("held-out ratio", _env.heldout_ratio);
  Env::plog("held-out ratings (ask)", s);
  Env::plog("held-out ratings (got)", _test_map.size());
  s = _env.validation_ratio * _ratings.nratings();
  set_validation_sample(s);
  Env::plog("validation ratio", _env.validation_ratio);
  Env::plog("validation ratings (ask)", s);
  Env::plog("validation ratings (got)", _validation_map.size());
}

void
GAPRec::set_test_sample(int s)
{
  int n = 0;
  while (n < s) {
    Rating r;
    get_random_rating1(r);
    _test_ratings.push_back(r);
    _test_map[r] = true;
    n++;
  }
  FILE *hef = fopen(Env::file_str("/test-ratings.txt").c_str(), "w");  
  fprintf(hef, "%s\n", ratingslist_s(_test_map).c_str());
  fclose(hef);
}

void
GAPRec::set_validation_sample(int s)
{
  int n = 0;
  while (n < s) {
    Rating r;
    get_random_rating1(r);
    if (is_test_rating(r))
      continue;
    _validation_ratings.push_back(r);
    _validation_map[r] = true;
    n++;
  }
  FILE *hef = fopen(Env::file_str("/validation-ratings.txt").c_str(), "w");
  fprintf(hef, "%s\n", ratingslist_s(_validation_map).c_str());
  fclose(hef);
}


string
GAPRec::ratingslist_s(SampleMap &mp)
{
  ostringstream sa;
  for (SampleMap::const_iterator i = mp.begin(); i != mp.end(); ++i) {
    const Rating &p = i->first;
    const IDMap &movies = _ratings.seq2movie();
    const IDMap &users = _ratings.seq2user();
    IDMap::const_iterator ni = users.find(p.first);
    IDMap::const_iterator mi = movies.find(p.second);
    sa << ni->second << "\t" << mi->second << "\n";
  }
  return sa.str();
}

void
GAPRec::initialize()
{
  double **ad = _acurr.data();
  double **bd = _bcurr.data();
  double **cd = _ccurr.data();
  double **dd = _dcurr.data();
  
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      ad[i][k] = _a + 0.01 * gsl_rng_uniform(_r);
      bd[i][k] = _b + 0.1 * gsl_rng_uniform(_r);
    }

  for (uint32_t i = 0; i < _m; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      cd[i][k] = _c + 0.01 * gsl_rng_uniform(_r);
      dd[i][k] = _d + 0.1 * gsl_rng_uniform(_r);
    }
  update_global_state();
}

void
GAPRec::update_global_state()
{
  set_gamma_exp1(_acurr, _bcurr, _Etheta);
  set_gamma_exp2(_acurr, _bcurr, _Elogtheta);
  set_gamma_exp1(_ccurr, _dcurr, _Ebeta);
  set_gamma_exp2(_ccurr, _dcurr, _Elogbeta);
  set_etheta_sum();
  set_ebeta_sum();
  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
}

void
GAPRec::set_etheta_sum()
{
  _etheta_sum.zero();
  const double **etheta  = _Etheta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    for (uint32_t n = 0; n < _n; ++n)
      _etheta_sum[k] += etheta[n][k];
  debug("etheta sum set to %s\n", _etheta_sum.s().c_str());
}

void
GAPRec::set_ebeta_sum()
{
  _ebeta_sum.zero();
  const double **ebeta  = _Ebeta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    for (uint32_t m = 0; m < _m; ++m)
      _ebeta_sum[k] += ebeta[m][k];
  debug("ebeta sum set to %s\n", _ebeta_sum.s().c_str());
}


void
GAPRec::set_to_prior_users(Matrix &a, Matrix &b)
{
  double **ad = a.data();
  double **bd = b.data();
  
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      ad[i][k] = _a;
      bd[i][k] = _b;
    }
}

void
GAPRec::set_to_prior_movies(Matrix &c, Matrix &d)
{
  double **cd = c.data();
  double **dd = d.data();
  
  for (uint32_t i = 0; i < _m; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      cd[i][k] = _c;
      dd[i][k] = _d;
    }
}

void
GAPRec::batch_infer()
{
  info("gaprec initialization done\n");

  initialize();
  approx_log_likelihood();
  fflush(stdout);

  while (1) {
    
    Array phi(_k);
    for (uint32_t n = 0; n < _n; ++n) {
      
      const double  **elogtheta = _Elogtheta.const_data();
      const double  **elogbeta = _Elogbeta.const_data();
      const vector<uint32_t> *movies = _ratings.get_movies(n);
      
      for (uint32_t j = 0; j < movies->size(); ++j) {
	uint32_t m = (*movies)[j];
	
	yval_t y = _ratings.r(n,m);
	assert (y >= 0);

	Rating r(n,m);
	if (!rating_ok(r))
	  continue;

	phi.zero();
	for (uint32_t k = 0; k < _k; ++k)
	  phi[k] = elogtheta[n][k] + elogbeta[m][k];
	phi.lognormalize();
	phi.scale(y);

	_anext.add_slice(n, phi);
	_cnext.add_slice(m, phi);
      }
      _bnext.add_slice(n, _ebeta_sum);
    }
    for (uint32_t m = 0; m < _m; ++m)
      _dnext.add_slice(m, _etheta_sum);

    _acurr.reset(_anext);
    _bcurr.reset(_bnext);
    _ccurr.reset(_cnext);
    _dcurr.reset(_dnext);

    update_global_state();

    printf("\r iteration %d", _iter);
    fflush(stdout);

    if (_iter % _env.reportfreq == 0) {
      approx_log_likelihood();
      validation_likelihood();
      test_likelihood();
    }
    _iter++;
  }
}

void
GAPRec::optimize_user_shape_parameters(uint32_t n)
{
  Array old(_k), curr(_k), v(_k);
  double **acurrd = _acurr.data();
  
  Matrix cphi(_m, _k);
  for (uint32_t i = 0; i < _env.online_iterations; ++i)  {
    for (uint32_t k = 0; k < _k; ++k)
      old[k] = acurrd[n][k];
    
    optimize_user_shape_parameters_helper(n, cphi);
    
    for (uint32_t k = 0; k < _k; ++k)
      curr[k] = acurrd[n][k];    
    
    sub(curr, old, v);
    if (v.abs_mean() < _env.meanchangethresh)
      break;
  }
  double **cnextd = _cnext.data();
  for (uint32_t m = 0; m < _m; ++m)
    for (uint32_t k = 0; k < _k; ++k)
      cnextd[m][k] += cphid[m][k];
}

void
GAPRec::optimize_user_shape_parameters_helper(uint32_t n, 
                                              Matrix &cphi)
{  
  const double  **elogtheta = _Elogtheta.const_data();
  const double  **elogbeta = _Elogbeta.const_data();
  const vector<uint32_t> *movies = _ratings.get_movies(n);
  
  cphi.zero();
  for (uint32_t j = 0; j < movies->size(); ++j) {
    uint32_t m = (*movies)[j];
    
    yval_t y = _ratings.r(n,m);
    assert (y >= 0);
    
    Rating r(n,m);
    if (!rating_ok(r))
      continue;
    
    phi.zero();
    for (uint32_t k = 0; k < _k; ++k)
      phi[k] = elogtheta[n][k] + elogbeta[m][k];
    phi.lognormalize();
    phi.scale(y);
    
    _anext.add_slice(n, phi);
    cphi.add_slice(m, phi);
  }

  double **acurrd = _acurr.data();
  double **anextd = _anext.data();
  for (uint32_t k = 0; k < _k; ++k) {
    acurrd[n][k] = anextd[n][k];
    anextd[n][k] = _a;
  }

  set_gamma_exp1_idx(n, _acurr, _bcurr, _Etheta);
  set_gamma_exp2_idx(n, _acurr, _bcurr, _Elogtheta);
}

void
GAPRec::optimize_user_rate_parameters(uint32_t n)
{
  double **bcurrd = _bcurr.data();
  double **bnextd = _bnext.data();
  _bnext.add_slice(n, _ebeta_sum);
  for (uint32_t k = 0; k < _k; ++k) {
    bcurrd[n][k] = bnextd[n][k];
    bnextd[n][k] = _b;
  }
}

void
GAPRec::recompute_etheta_sum(const UserMap &sampled_users, 
                             const Array &etheta_sum_old)
{
  const double **etheta  = _Etheta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _etheta_sum[k] -= etheta_sum_old[k];
  
  for (UserMap::const_iterator itr = sampled_users.begin();
       itr != sampled_users.end(); ++itr) {
    uint32_t user = itr->first;    
    for (uint32_t k = 0; k < _k; ++k)
      _etheta_sum[k] += etheta[user][k];
  }
}

void
GAPRec::infer()
{
  initialize();
  approx_log_likelihood();
  
  const double **etheta  = _Etheta.const_data();
  Array phi(_k);
  while (1) {
    Usermap sampled_users;
    do {
      uint32_t n = gsl_rng_uniform_int(_r, _n);
      UserMap::const_iterator itr = sampled_users.find(n);
      if (itr == sampled_users.end()) {
        _anext.set_elements(n, _a);
        _bnext.set_elements(n, _b);
        _etheta_sum.zero(); 
        set_to_prior_movies(_cnext, _dnext);
        
        sampled_users[n] = true;
        set_gamma_exp1_idx(n, _acurr, _bcurr, _Etheta);
        set_gamma_exp2_idx(n, _acurr, _bcurr, _Elogtheta);
      }
    } while (sampled_users.size() < _env.mini_batch_size);
    
    for (UserMap::const_iterator itr = sampled_users.begin();
         itr != sampled_users.end(); ++itr) {
      uint32_t user = itr->first;
      // update rates before shape
      optimize_user_rate_parameters(user);
      optimize_user_shape_parameters(user);
      for (uint32_t k = 0; k < _k; ++k)
        _etheta_sum[k] += etheta[user][k];
    }

    for (uint32_t m = 0; m < _m; ++m)
      _dnext.add_slice(m, _etheta_sum);

    double scale = _n / _sampled_users.size();
    
    // stochastic gradient update of ccurr and dcurr
    for (uint32_t m = 0; m < _m; ++m) {
      ccurrd[m][k] = (1 - rho) * ccurrd[m][k] + \
        rho * (_c + scale * cnextd[m][k]);
      dcurrd[m][k] = (1 - rho) * dcurrd[m][k] + \
        rho * (_d + scale * dnextd[m][k]);
    }
    set_gamma_exp1(_ccurr, _dcurr, _Ebeta);
    set_gamma_exp2(_ccurr, _dcurr, _Elogbeta);
    set_ebeta_sum();
    
    printf("\r iteration %d", _iter);
    fflush(stdout);
    
    if (_iter % _env.reportfreq == 0) {
      approx_log_likelihood();
      test_likelihood();
      validation_likelihood();
    }
    _iter++;
}

void
GAPRec::set_gamma_exp1(const Matrix &a, const Matrix &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double ** const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      assert(bd[i][j]);
      vd[i][j] = ad[i][j] / bd[i][j];
    }
}

void
GAPRec::set_gamma_exp1_idx(uint32_t u, 
			   const Matrix &a, const Matrix &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double ** const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t j = 0; j < b.n(); ++j) {
    assert(bd[u][j]);
    vd[u][j] = ad[u][j] / bd[u][j];
  }
}

void
GAPRec::set_gamma_exp2(const Matrix &a, const Matrix &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double ** const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      assert(bd[i][j]);
      vd[i][j] = gsl_sf_psi(ad[i][j]) - log(bd[i][j]);
    }
}

void
GAPRec::set_gamma_exp2_idx(uint32_t u, 
			   const Matrix &a, const Matrix &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double ** const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t j = 0; j < b.n(); ++j) {
    assert(bd[u][j]);
    vd[u][j] = gsl_sf_psi(ad[u][j]) - log(bd[u][j]);
  }
}

void
GAPRec::approx_log_likelihood()
{
  return; // XXX

  const double ** etheta = _Etheta.const_data();
  const double ** elogtheta = _Elogtheta.const_data();
  const double ** ebeta = _Ebeta.const_data();
  const double ** elogbeta = _Elogbeta.const_data();
  
  const double ** const ad = _acurr.const_data();
  const double ** const bd = _bcurr.const_data();
  const double ** const cd = _ccurr.const_data();
  const double ** const dd = _dcurr.const_data();

  debug("Etheta = %s\n", _Etheta.s().c_str());
  debug("Elogtheta = %s\n", _Elogtheta.s().c_str());
  debug("Ebeta = %s\n", _Ebeta.s().c_str());
  debug("Elogbeta = %s\n", _Elogbeta.s().c_str());
  debug("a = %s\n", _acurr.s().c_str());
  debug("b = %s\n", _bcurr.s().c_str());
  debug("c = %s\n", _ccurr.s().c_str());
  debug("d = %s\n", _dcurr.s().c_str());

  double s = .0;

  Array phi(_k);
  for (uint32_t n = 0; n < _n; ++n) {
    
    const vector<uint32_t> *movies = _ratings.get_movies(n);
    
    for (uint32_t j = 0; j < movies->size(); ++j) {
      uint32_t m = (*movies)[j];
      
      yval_t y = _ratings.r(n,m);
      
      Rating r(n,m);
      if (!rating_ok(r))
	continue;
      
      phi.zero();
      for (uint32_t k = 0; k < _k; ++k)
	phi[k] = elogtheta[n][k] + elogbeta[m][k];
      phi.lognormalize();

      double v = .0;
      for (uint32_t k = 0; k < _k; ++k) 
	v += y * phi[k] * (elogtheta[n][k] + elogbeta[m][k] - log(phi[k]));
      s += v;
      
      for (uint32_t k = 0; k < _k; ++k)
	s -= etheta[n][k] * ebeta[m][k];
    }
  }
  
  for (uint32_t n = 0; n < _n; ++n)  {
    for (uint32_t k = 0; k < _k; ++k) {
      s += _a * log(_b) + (_a - 1) * elogtheta[n][k];
      s -= _b * etheta[n][k] + gsl_sf_lngamma(_a);
    }
    for (uint32_t k = 0; k < _k; ++k) {
      s -= ad[n][k] * log(bd[n][k]) + (ad[n][k] - 1) * elogtheta[n][k];
      s += bd[n][k] * etheta[n][k] + gsl_sf_lngamma(ad[n][k]);
    }
  }

  for (uint32_t m = 0; m < _m; ++m)  {
    for (uint32_t k = 0; k < _k; ++k) {
      s += _c * log(_d) + (_c - 1) * elogbeta[m][k];
      s -= _d * ebeta[m][k] + gsl_sf_lngamma(_c);
    }
    for (uint32_t k = 0; k < _k; ++k) {
      s -= cd[m][k] * log(dd[m][k]) + (cd[m][k] - 1) * elogbeta[m][k];
      s += dd[m][k] * ebeta[m][k] + gsl_sf_lngamma(cd[m][k]);
    }
  }
  fprintf(_af, "%.5f\n", s);
  fflush(_af);
}


void
GAPRec::test_likelihood()
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (SampleMap::const_iterator i = _test_map.begin();
       i != _test_map.end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    yval_t r = _ratings.r(n,m);
    double u = pair_likelihood(n,m,r);
    s += u;
    k += 1;
    info("n = %d, m  = %d, r = %d, u = %.5f\n", n, m, r, u);
  }
  info("s = %.5f\n", s);
  fprintf(_hf, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(_hf);
}

void
GAPRec::validation_likelihood()
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (SampleMap::const_iterator i = _validation_map.begin();
       i != _validation_map.end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    yval_t r = _ratings.r(n,m);
    double u = pair_likelihood(n,m,r);
    s += u;
    k += 1;
    info("n = %d, m  = %d, r = %d, u = %.5f\n", n, m, r, u);
  }
  info("s = %.5f\n", s);
  fprintf(_vf, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(_vf);
}

uint32_t
GAPRec::factorial(uint32_t n)  const
{ 
  return n <= 1 ? 1 : (n * factorial(n-1));
} 

double
GAPRec::pair_likelihood(uint32_t p, uint32_t q, yval_t y) const
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += etheta[p][k] * ebeta[q][k];
  if (s < 1e-30)
    s = 1e-30;
  info("%d, %d, s = %f, f(y) = %ld\n", p, q, s, factorial(y));
  return y * log(s) - s - log (factorial(y));
}

void
GAPRec::get_random_rating1(Rating &r) const
{
  do {
    uint32_t user = gsl_rng_uniform_int(_r, _n);
    uint32_t movie = gsl_rng_uniform_int(_r, _m);    

    if (_ratings.r(user, movie) > 0) {
      r.first = user;
      r.second = movie;
      break;
    }
  } while (1);  
}

void
GAPRec::get_random_rating2(Rating &r) const
{
  // note: this first picks a random node, then picks a movie
  // so it is not a uniform sample of movies
  do {
    uint32_t user = gsl_rng_uniform_int(_r, _n);
    const vector<uint32_t> *movies = _ratings.get_movies(user);
    
    if (movies && movies->size() > 0) {
      uint32_t m = gsl_rng_uniform_int(_r, movies->size());
      uint32_t mov = movies->at(m);
      assert (_ratings.r(user, mov) > 0);
      
      r.first = user;
      r.second = mov;
      break;
    }
  } while (1);
}

