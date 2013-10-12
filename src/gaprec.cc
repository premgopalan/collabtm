#include "gaprec.hh"

GAPRec::GAPRec(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _n(env.n), _m(env.m), _k(env.k),
    _iter(0),
    _a(env.a), _b(env.b),
    _c(env.c), _d(env.d),
    _start_time(time(0)),
    _acurr(_n,_k), _bcurr(_k),
    _ccurr(_m,_k), _dcurr(_k),
    _anext(_n,_k), _bnext(_k),
    _cnext(_m,_k), _dnext(_k),
    _ucurr(_n), _icurr(_m),
    _unext(_n), _inext(_m),
    _phi(_k),
    _Elogtheta(_n,_k), _Etheta(_n,_k),
    _Elogbeta(_m,_k), _Ebeta(_m,_k),
    _Elogu(_n), _Elogi(_m),
    _Eu(_n), _Ei(_m),
    _etheta_sum(_k), _etheta_sum_old(_k),
    _ebeta_sum(_k),  _ebeta_sum_old(_k),
    _rho(_m),
    _tau0(1024), _kappa(0.5),
    _nh(0), _prev_h(.0),
    _save_ranking_file(false),
    _itemc(_m),
    _use_rate_as_score(true),
    _topN_by_user(100)
{
  _itemc.zero();
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed)
    gsl_rng_set(_r, _env.seed);
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
  _pf = fopen(Env::file_str("/precision.txt").c_str(), "w");
  if (!_pf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  if (_env.mode == Env::CREATE_TRAIN_TEST_SETS) {
    init_heldout();
  } else
    load_validation_and_test_sets();
}

GAPRec::~GAPRec()
{
  fclose(_hf);
  fclose(_vf);
  fclose(_af);
  fclose(_pf);
}

// Note: test set is loaded only to compute precision
void
GAPRec::load_validation_and_test_sets()
{
  char buf[4096];
  sprintf(buf, "%s/validation.tsv", _env.datfname.c_str());
  FILE *validf = fopen(buf, "r");
  assert(validf);
  _ratings.read_generic(validf, &_validation_map);
  fclose(validf);

  sprintf(buf, "%s/test.tsv", _env.datfname.c_str());
  FILE *testf = fopen(buf, "r");
  assert(testf);
  _ratings.read_generic(testf, &_test_map);
  fclose(testf);
  printf("+ loaded validation and test sets from %s\n", _env.datfname.c_str());
  fflush(stdout);
  Env::plog("test ratings", _test_map.size());
  Env::plog("validation ratings", _validation_map.size());
}

void
GAPRec::init_heldout()
{
  int s = _env.heldout_ratio * _ratings.nratings();
  if (_env.dataset == Env::NETFLIX || 
      _env.dataset == Env::MENDELEY || 
      _env.dataset == Env::ECHONEST) {
    set_test_sample(s);
  } 
  Env::plog("held-out ratio", _env.heldout_ratio);
  Env::plog("held-out ratings (ask)", s);
  Env::plog("held-out ratings (got)", _test_map.size());
  s = _env.validation_ratio * _ratings.nratings();
  set_validation_sample(s);
  Env::plog("validation ratio", _env.validation_ratio);
  Env::plog("validation ratings (ask)", s);
  Env::plog("validation ratings (got)", _validation_map.size());
  set_training_sample();
}

void
GAPRec::set_training_sample()
{
  const IDMap &movies = _ratings.seq2movie();
  const IDMap &users = _ratings.seq2user();

  FILE *trainf = fopen(Env::file_str("/train.tsv").c_str(), "w");
  for (uint32_t n = 0; n < _n; ++n) {
    const vector<uint32_t> *movs = _ratings.get_movies(n);
    for (uint32_t j = 0; j < movs->size(); ++j) {
      uint32_t m = (*movs)[j];
      Rating r(n,m);
      if (!rating_ok(r))
	continue;

      uint32_t v = _ratings.r(n,m);
      IDMap::const_iterator ni = users.find(n);
      IDMap::const_iterator mi = movies.find(m);
      fprintf(trainf, "%d\t%d\t%d\n", ni->second, mi->second, v);
    }
    fflush(trainf);
  }
  fclose(trainf);
}

void
GAPRec::set_test_sample(int s)
{
  int n = 0;
  while (n < s) {
    Rating r;
    get_random_rating3(r);
    CountMap::const_iterator itr = _test_map.find(r);
    if (itr == _test_map.end()) {
      _test_map[r] = true;
      n++;
    }
  }
  FILE *hef = fopen(Env::file_str("/test.tsv").c_str(), "w");  
  write_sample(hef, _test_map);
  fclose(hef);
}

void
GAPRec::set_validation_sample(int s)
{
  int n = 0;
  while (n < s) {
    Rating r;
    get_random_rating3(r);
    if (is_test_rating(r))
      continue;
    CountMap::const_iterator itr = _validation_map.find(r);
    if (itr == _validation_map.end()) {
      _validation_map[r] = 1;
      n++;
    }
  }
  FILE *hef = fopen(Env::file_str("/validation.tsv").c_str(), "w");
  write_sample(hef, _validation_map);
  fclose(hef);
}

void
GAPRec::write_sample(FILE *f, CountMap &mp)
{
  for (CountMap::const_iterator i = mp.begin(); i != mp.end(); ++i) {
    const Rating &p = i->first;
    uint32_t v = _ratings.r(p.first, p.second);
    const IDMap &movies = _ratings.seq2movie();
    const IDMap &users = _ratings.seq2user();
    IDMap::const_iterator ni = users.find(p.first);
    IDMap::const_iterator mi = movies.find(p.second);
    fprintf(f, "%d\t%d\t%d\n", ni->second, mi->second, v);
  }
  fflush(f);
}


void
GAPRec::initialize()
{
  double **ad = _acurr.data();
  double *bd = _bcurr.data();
  double **cd = _ccurr.data();
  double *dd = _dcurr.data();
  
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k) 
      ad[i][k] = _a + 0.01 * gsl_rng_uniform(_r);

  for (uint32_t k = 0; k < _k; ++k)
    bd[k] = _b + 0.1 * gsl_rng_uniform(_r);
  
  for (uint32_t i = 0; i < _m; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      cd[i][k] = _c + 0.01 * gsl_rng_uniform(_r);
  
  for (uint32_t k = 0; k < _k; ++k)       
    dd[k] = _d + 0.1 * gsl_rng_uniform(_r);
      
  set_gamma_exp_init(_acurr, _Etheta, _Elogtheta, _b);
  set_gamma_exp_init(_ccurr, _Ebeta, _Elogbeta, _d);
  set_etheta_sum();
  set_ebeta_sum();
  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
}

void
GAPRec::update_global_state()
{
  set_gamma_exp(_acurr, _bcurr, _Etheta, _Elogtheta);
  set_gamma_exp(_ccurr, _dcurr, _Ebeta, _Elogbeta);
  if (_env.batch) {
    set_etheta_sum();
    set_ebeta_sum();
  } else {
    set_etheta_sum2();
    set_ebeta_sum2();
  }
  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
}

void
GAPRec::initialize_bias()
{
  double **ad = _acurr.data();
  double *bd = _bcurr.data();
  double **cd = _ccurr.data();
  double *dd = _dcurr.data();
  
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t k = 0; k < _k; ++k) 
      ad[i][k] = _a + 0.01 * gsl_rng_uniform(_r);
    _ucurr[i] = _a + 0.01 * gsl_rng_uniform(_r);
  }

  for (uint32_t k = 0; k < _k; ++k)
    bd[k] = _b + 0.1 * gsl_rng_uniform(_r);
  
  for (uint32_t i = 0; i < _m; ++i) {
    for (uint32_t k = 0; k < _k; ++k) 
      cd[i][k] = _c + 0.01 * gsl_rng_uniform(_r);
    _icurr[i] = _c + 0.01 * gsl_rng_uniform(_r);
  }
  
  for (uint32_t k = 0; k < _k; ++k)       
    dd[k] = _d + 0.1 * gsl_rng_uniform(_r);

  set_gamma_exp_init(_acurr, _Etheta, _Elogtheta, _b);
  set_gamma_exp_init(_ccurr, _Ebeta, _Elogbeta, _d);
  set_gamma_exp_bias(_ucurr, _b + _m, _Eu, _Elogu);
  set_gamma_exp_bias(_icurr, _d + _n, _Ei, _Elogi);
  set_etheta_sum();
  set_ebeta_sum();
  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
  set_to_prior_biases();
}

void
GAPRec::set_to_prior_biases()
{
  _unext.set_elements(_a);
  _inext.set_elements(_c);
}

void
GAPRec::update_global_state_bias()
{
  set_gamma_exp(_acurr, _bcurr, _Etheta, _Elogtheta);
  set_gamma_exp(_ccurr, _dcurr, _Ebeta, _Elogbeta);
  set_gamma_exp_bias(_ucurr, _b + _m, _Eu, _Elogu);
  set_gamma_exp_bias(_icurr, _d + _n, _Ei, _Elogi);
  if (_env.batch) {
    set_etheta_sum();
    set_ebeta_sum();
  } else {
    set_etheta_sum2();
    set_ebeta_sum2();
  }
  set_to_prior_users(_anext, _bnext);
  set_to_prior_movies(_cnext, _dnext);
  set_to_prior_biases();
}


void
GAPRec::set_etheta_sum()
{
  _etheta_sum.zero();
  const double **etheta  = _Etheta.const_data();
  for (uint32_t k = 0; k < _k; ++k) {
    for (uint32_t n = 0; n < _n; ++n)
      _etheta_sum[k] += etheta[n][k];
  }
  debug("etheta sum set to %s\n", _etheta_sum.s().c_str());
}

void
GAPRec::set_ebeta_sum()
{
  _ebeta_sum.zero();
  const double **ebeta  = _Ebeta.const_data();
  for (uint32_t k = 0; k < _k; ++k) {
    for (uint32_t m = 0; m < _m; ++m)
      _ebeta_sum[k] += ebeta[m][k];
  }
  debug("ebeta sum set to %s\n", _ebeta_sum.s().c_str());
}

void
GAPRec::set_etheta_sum2()
{
  _etheta_sum.zero();
  const double **etheta  = _Etheta.const_data();
  for (uint32_t k = 0; k < _k; ++k) {
    _etheta_sum[k] = _b;
    for (uint32_t n = 0; n < _n; ++n)
      _etheta_sum[k] += etheta[n][k];
  }
  debug("etheta sum set to %s\n", _etheta_sum.s().c_str());
}

void
GAPRec::set_ebeta_sum2()
{
  _ebeta_sum.zero();
  const double **ebeta  = _Ebeta.const_data();
  for (uint32_t k = 0; k < _k; ++k) {
    _ebeta_sum[k] = _d;
    for (uint32_t m = 0; m < _m; ++m)
      _ebeta_sum[k] += ebeta[m][k];
  }
  debug("ebeta sum set to %s\n", _ebeta_sum.s().c_str());
}

void
GAPRec::set_to_prior_users(Matrix &a, Array &b)
{
  a.set_elements(_a);
  b.set_elements(_b);
}

void
GAPRec::set_to_prior_movies(Matrix &c, Array &d)
{
  c.set_elements(_c);
  d.set_elements(_d);
}

void
GAPRec::batch_infer()
{
  info("gaprec initialization done\n");

  initialize();
  approx_log_likelihood();
  fflush(stdout);

  Array phi(_k);
  while (1) {
    uint32_t nr = 0;
    
    for (uint32_t n = 0; n < _n; ++n) {
      
      const double  **elogtheta = _Elogtheta.const_data();
      const double  **elogbeta = _Elogbeta.const_data();
      const vector<uint32_t> *movies = _ratings.get_movies(n);
      
      for (uint32_t j = 0; j < movies->size(); ++j) {
	uint32_t m = (*movies)[j];
	yval_t y = _ratings.r(n,m);
	
	phi.zero();
	for (uint32_t k = 0; k < _k; ++k)
	  phi[k] = elogtheta[n][k] + elogbeta[m][k];
	phi.lognormalize();
	if (y > 1)
	  phi.scale(y);
	
	_anext.add_slice(n, phi);
	_cnext.add_slice(m, phi);
	nr++;
	if (nr % 100000 == 0)
	  lerr("iter:%d ratings:%d total:%d frac:%.3f", 
	       _iter, nr, _ratings.nratings(), 
	       (double)nr / _ratings.nratings());
      }
    }
    lerr("updating bnext and dnext");
    _bnext.add_to(_ebeta_sum);
    _dnext.add_to(_etheta_sum);
    lerr("done updating dnext");

    _acurr.swap(_anext);
    _bcurr.swap(_bnext);
    _ccurr.swap(_cnext);
    _dcurr.swap(_dnext);

    lerr("done swapping");

    update_global_state();

    lerr("done updating global state");

    printf("\r iteration %d", _iter);
    fflush(stdout);
    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      auc();
      save_model();
      _env.save_state_now = false;
    }

    if (_iter % _env.reportfreq == 0) {
      lerr("computing validation likelihood");
      validation_likelihood();
      lerr("done computing validation likelihood");
      lerr("saving model");
      save_model();
      lerr("done saving model");
      auc();
    }
    _iter++;
  }
}


void
GAPRec::batch_infer_bias()
{
  initialize_bias();
  approx_log_likelihood();
  fflush(stdout);

  Array phi(_k+2);
  while (1) {
    uint32_t nr = 0;
    
    for (uint32_t n = 0; n < _n; ++n) {
      
      const double  **elogtheta = _Elogtheta.const_data();
      const double  **elogbeta = _Elogbeta.const_data();
      const vector<uint32_t> *movies = _ratings.get_movies(n);
      
      for (uint32_t j = 0; j < movies->size(); ++j) {
	uint32_t m = (*movies)[j];
	yval_t y = _ratings.r(n,m);
	
	phi.zero();
	for (uint32_t k = 0; k < _k; ++k)
	  phi[k] = elogtheta[n][k] + elogbeta[m][k];
	phi[_k] = _Elogu[n];
	phi[_k+1] = _Elogi[m];
	phi.lognormalize();
	if (y > 1)
	  phi.scale(y);
	
	_anext.add_slice(n, phi);
	_cnext.add_slice(m, phi);
	_unext[n] += phi[_k];
	_inext[m] += phi[_k+1];
	nr++;
	if (nr % 100000 == 0)
	  lerr("iter:%d ratings:%d total:%d frac:%.3f", 
	       _iter, nr, _ratings.nratings(), 
	       (double)nr / _ratings.nratings());
      }
    }
    lerr("updating bnext and dnext");
    _bnext.add_to(_ebeta_sum);
    _dnext.add_to(_etheta_sum);

    lerr("done updating dnext");
    _acurr.swap(_anext);
    _bcurr.swap(_bnext);
    _ccurr.swap(_cnext);
    _dcurr.swap(_dnext);
    _ucurr.swap(_unext);
    _icurr.swap(_inext);

    lerr("done swapping");
    update_global_state_bias();

    lerr("done updating global state");
    printf("\r iteration %d", _iter);
    fflush(stdout);

    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      save_model();
      auc(true);
      _env.save_state_now = false;
    }

    if (_iter % _env.reportfreq == 0) {
      lerr("computing validation likelihood");
      validation_likelihood(true);
      lerr("done computing validation likelihood");
      lerr("saving model");
      save_model();
      lerr("done saving model");
      auc(true);
    }
    _iter++;
  }
}


void
GAPRec::optimize_user_shape_parameters2(uint32_t n)
{
  const double  **elogtheta = _Elogtheta.const_data();
  const double  **elogbeta = _Elogbeta.const_data();
  const vector<uint32_t> *movies = _ratings.get_movies(n);
  
  for (uint32_t j = 0; j < movies->size(); ++j) {
    uint32_t m = (*movies)[j];
    
    yval_t y = _ratings.r(n,m);
    assert (y >= 0);
    
    Rating r(n,m);
    if (!rating_ok(r))
      assert(0);
    
    _phi.zero();
    for (uint32_t k = 0; k < _k; ++k)
      _phi[k] = elogtheta[n][k] + elogbeta[m][k];
    _phi.lognormalize();
    _phi.scale(y);
    
    _anext.add_slice(n, _phi);
    _cnext.add_slice(m, _phi);
  }
  double **acurrd = _acurr.data();
  double **anextd = _anext.data();
  for (uint32_t k = 0; k < _k; ++k) {
    acurrd[n][k] = anextd[n][k];
    anextd[n][k] = _a;
  }
  set_gamma_exp1_array(n, _acurr, _ebeta_sum, _Etheta);
  set_gamma_exp2_array(n, _acurr, _ebeta_sum, _Elogtheta);
}

void
GAPRec::optimize_user_shape_parameters(uint32_t n)
{
  Array old(_k), curr(_k), v(_k);
  double **acurrd = _acurr.data();
  
  Matrix cphi(_m, _k);
  double **cphid = cphi.data();
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
  const vector<uint32_t> *movies = _ratings.get_movies(n);
  for (uint32_t j = 0; j < movies->size(); ++j) {
    uint32_t m = (*movies)[j];
    for (uint32_t k = 0; k < _k; ++k)
      cnextd[m][k] += cphid[m][k];
  }
}

void
GAPRec::optimize_user_shape_parameters_helper(uint32_t n, 
                                              Matrix &cphi)
{  
  const double  **elogtheta = _Elogtheta.const_data();
  const double  **elogbeta = _Elogbeta.const_data();
  const vector<uint32_t> *movies = _ratings.get_movies(n);
  
  cphi.zero();
  Array phi(_k);
  for (uint32_t j = 0; j < movies->size(); ++j) {
    uint32_t m = (*movies)[j];
    
    yval_t y = _ratings.r(n,m);
    assert (y >= 0);
    
    Rating r(n,m);
    if (!rating_ok(r))
      assert(0);
    
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
  
  set_gamma_exp1_array(n, _acurr, _ebeta_sum, _Etheta);
  set_gamma_exp2_array(n, _acurr, _ebeta_sum, _Elogtheta);
}


void
GAPRec::compute_etheta_sum(const UserMap &sampled_users)
{
  const double **etheta  = _Etheta.const_data();
  _etheta_sum_old.zero();
  for (UserMap::const_iterator itr = sampled_users.begin();
       itr != sampled_users.end(); ++itr) {
    uint32_t user = itr->first;    
    for (uint32_t k = 0; k < _k; ++k)
      _etheta_sum_old[k] += etheta[user][k];
  }
}

void
GAPRec::adjust_etheta_sum(const UserMap &sampled_users)
{
  const double **etheta  = _Etheta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _etheta_sum[k] -= _etheta_sum_old[k];
  
  for (UserMap::const_iterator itr = sampled_users.begin();
       itr != sampled_users.end(); ++itr) {
    uint32_t user = itr->first;    
    for (uint32_t k = 0; k < _k; ++k)
      _etheta_sum[k] += etheta[user][k];
  }
}


void
GAPRec::compute_ebeta_sum(const MovieMap &sampled_movies)
{
  const double **ebeta  = _Ebeta.const_data();
  _ebeta_sum_old.zero();
  for (UserMap::const_iterator itr = sampled_movies.begin();
       itr != sampled_movies.end(); ++itr) {
    uint32_t movie = itr->first;    
    for (uint32_t k = 0; k < _k; ++k)
      _ebeta_sum_old[k] += ebeta[movie][k];
  }
}

void
GAPRec::adjust_ebeta_sum(const MovieMap &sampled_movies)
{
  const double **ebeta  = _Ebeta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    _ebeta_sum[k] -= _ebeta_sum_old[k];
  
  for (MovieMap::const_iterator itr = sampled_movies.begin();
       itr != sampled_movies.end(); ++itr) {
    uint32_t movie = itr->first;    
    for (uint32_t k = 0; k < _k; ++k)
      _ebeta_sum[k] += ebeta[movie][k];
  }
}

void
GAPRec::infer()
{
  initialize();
  approx_log_likelihood();

  const double **etheta  = _Etheta.const_data();
  const double **ebeta  = _Ebeta.const_data();
  while (1) {
    //
    // local step
    //
    _sampled_users.clear();
    _sampled_movies.clear();
    do {
      uint32_t n = gsl_rng_uniform_int(_r, _n);
      
      UserMap::const_iterator itr = _sampled_users.find(n);
      if (itr == _sampled_users.end()) {
        _anext.set_elements(n, _a);
        _sampled_users[n] = true;
	
	const vector<uint32_t> *movies = _ratings.get_movies(n);
	for (uint32_t j = 0; j < movies->size(); ++j) {
	  uint32_t m = (*movies)[j];
	  _cnext.set_elements(m, _c);
	  _sampled_movies[m] = true;
	}
      }
    } while (_sampled_users.size() < _env.mini_batch_size);
    
    lerr("sampled %d users and %d movies", _sampled_users.size(), _sampled_movies.size());

    compute_ebeta_sum(_sampled_movies);
    compute_etheta_sum(_sampled_users);
    
    for (UserMap::const_iterator itr = _sampled_users.begin();
         itr != _sampled_users.end(); ++itr) {
      uint32_t user = itr->first;
      // update rates before shape
      optimize_user_shape_parameters2(user);
    }
    
    double scale_users = _n / _sampled_users.size();

    // stochastic gradient update of ccurr and dcurr
    double **ccurrd = _ccurr.data();
    double **cnextd = _cnext.data();

    for (MovieMap::const_iterator itr = _sampled_movies.begin();
         itr != _sampled_movies.end(); ++itr) {
      uint32_t m = itr->first;

      _rho[m] = pow(_tau0 + _itemc[m], -1 * _kappa);

      for (uint32_t k = 0; k < _k; ++k)
	ccurrd[m][k] = (1 - _rho[m]) * ccurrd[m][k] +	\
	  _rho[m] * (_c + scale_users * cnextd[m][k]);

      set_gamma_exp1_array(m, _ccurr, _etheta_sum, _Ebeta);
      set_gamma_exp2_array(m, _ccurr, _etheta_sum, _Elogbeta);
      _itemc[m]++;
    }

    adjust_etheta_sum(_sampled_users);
    adjust_ebeta_sum(_sampled_movies);

    if (_env.save_state_now) {
      lerr("Saving state at iteration %d duration %d secs", _iter, duration());
      auc();
      save_model();
      _env.save_state_now = false;
    }
    
    if (_iter % _env.reportfreq == 0) {
      lerr("iteration %d", _iter);
      validation_likelihood();
      if (_iter > 0 && _iter % 100 == 0) { // min. 100
	lerr("auc..");
	auc();
	lerr("done; saving model...");
	save_model();
	lerr("done");
      }
    }
    _iter++;
  }
}

void
GAPRec::set_gamma_exp(const Matrix &a, const Array &b, Matrix &v1, Matrix &v2)
{
  const double ** const ad = a.const_data();
  const double * const bd = b.const_data();
  double **vd1 = v1.data();
  double **vd2 = v2.data();
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      assert(bd[j]);
      vd1[i][j] = ad[i][j] / bd[j];
      vd2[i][j] = gsl_sf_psi(ad[i][j]) - log(bd[j]);
    }
}

void
GAPRec::set_gamma_exp_bias(const Array &u, double v, Array &w1, Array &w2)
{
  for (uint32_t i = 0; i < u.n(); ++i) {
    assert(v);
    w1[i] = u[i] / v;
    w2[i] = gsl_sf_psi(u[i]) - log(v);
  }
}

void
GAPRec::set_gamma_exp_init(const Matrix &a, Matrix &v1, Matrix &v2, double v)
{
  const double ** const ad = a.const_data();
  double **vd1 = v1.data();
  double **vd2 = v2.data();
  
  Array b(_k);
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      b[j] = v + 0.1 * gsl_rng_uniform(_r);
      assert(b[j]);
      vd1[i][j] = ad[i][j] / b[j];
      vd2[i][j] = gsl_sf_psi(ad[i][j]) - log(b[j]);
    }
}

void
GAPRec::set_gamma_exp1(const Matrix &a, const Array &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double * const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      assert(bd[j]);
      vd[i][j] = ad[i][j] / bd[j];
    }
}

void
GAPRec::set_gamma_exp2(const Matrix &a, const Array &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double * const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t i = 0; i < a.m(); ++i)
    for (uint32_t j = 0; j < b.n(); ++j) {
      assert(bd[j]);
      vd[i][j] = gsl_sf_psi(ad[i][j]) - log(bd[j]);
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
GAPRec::set_gamma_exp1_array(uint32_t u, 
			     const Matrix &a, const Array &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double * const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t j = 0; j < b.n(); ++j) {
    assert(bd[j]);
    vd[u][j] = ad[u][j] / bd[j];
  }
}

void
GAPRec::set_gamma_exp2_array(uint32_t u, 
			     const Matrix &a, const Array &b, Matrix &v)
{
  const double ** const ad = a.const_data();
  const double * const bd = b.const_data();
  double **vd = v.data();
  for (uint32_t j = 0; j < b.n(); ++j) {
    assert(bd[j]);
    vd[u][j] = gsl_sf_psi(ad[u][j]) - log(bd[j]);
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
  const double * const bd = _bcurr.const_data();
  const double ** const cd = _ccurr.const_data();
  const double * const dd = _dcurr.const_data();

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
	assert(0);
      
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
      s -= ad[n][k] * log(bd[k]) + (ad[n][k] - 1) * elogtheta[n][k];
      s += bd[k] * etheta[n][k] + gsl_sf_lngamma(ad[n][k]);
    }
  }

  for (uint32_t m = 0; m < _m; ++m)  {
    for (uint32_t k = 0; k < _k; ++k) {
      s += _c * log(_d) + (_c - 1) * elogbeta[m][k];
      s -= _d * ebeta[m][k] + gsl_sf_lngamma(_c);
    }
    for (uint32_t k = 0; k < _k; ++k) {
      s -= cd[m][k] * log(dd[k]) + (cd[m][k] - 1) * elogbeta[m][k];
      s += dd[k] * ebeta[m][k] + gsl_sf_lngamma(cd[m][k]);
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
  for (CountMap::const_iterator i = _test_map.begin();
       i != _test_map.end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    yval_t r = _ratings.r(n,m);
    double u = pair_likelihood(n,m,r);
    s += u;
    k += 1;
    info("test: n = %d, m  = %d, r = %d, u = %.5f\n", n, m, r, u);
  }
  info("s = %.5f\n", s);
  fprintf(_hf, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(_hf);
}


double
GAPRec::link_prob(uint32_t user, uint32_t movie) const
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += etheta[user][k] * ebeta[movie][k];
  
  if (_use_rate_as_score)
    return s;

  if (s < 1e-30)
    s = 1e-30;
  double prob_zero = exp(-s);
  return 1 - prob_zero;
}

double
GAPRec::link_prob_bias(uint32_t user, uint32_t movie) const
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += etheta[user][k] * ebeta[movie][k];
  s += _Eu[user] + _Ei[movie];

  if (_use_rate_as_score)
    return s;

  if (s < 1e-30)
    s = 1e-30;
  double prob_zero = exp(-s);
  return 1 - prob_zero;
}

void
GAPRec::auc(bool bias)
{
  double mhits10 = 0, mhits100 = 0;
  uint32_t total_users = 0;
  FILE *f = 0;
  if (_save_ranking_file)
    f = fopen(Env::file_str("/ranking.tsv").c_str(), "w");
  
  if (!_save_ranking_file) {
    _sampled_users.clear();
    do {
      uint32_t n = gsl_rng_uniform_int(_r, _n);
      _sampled_users[n] = true;
    } while (_sampled_users.size() < 1000 && _sampled_users.size() < _n / 2);
  }
  
  KVArray mlist(_m);
  for (UserMap::const_iterator itr = _sampled_users.begin();
       itr != _sampled_users.end(); ++itr) {
    uint32_t n = itr->first;
    
    for (uint32_t m = 0; m < _m; ++m) {
      if (_ratings.r(n,m) > 0) { // skip training
	mlist[m].first = m;
	mlist[m].second = .0;
	continue;
      }
      double u = bias ? link_prob_bias(n, m) : link_prob(n,m);
      mlist[m].first = m;
      mlist[m].second = u;
    }
    uint32_t hits10 = 0, hits100 = 0;
    mlist.sort_by_value();
    for (uint32_t j = 0; j < mlist.size() && j < _topN_by_user; ++j) {
      KV &kv = mlist[j];
      uint32_t m = kv.first;
      double pred = kv.second;
      Rating r(n, m);

      uint32_t m2 = 0, n2 = 0;
      if (_save_ranking_file) {
	IDMap::const_iterator it = _ratings.seq2user().find(n);
	assert (it != _ratings.seq2user().end());
	
	IDMap::const_iterator mt = _ratings.seq2movie().find(m);
	if (mt == _ratings.seq2movie().end())
	  continue;
      
	m2 = mt->second;
	n2 = it->second;
      }

      CountMap::const_iterator itr = _test_map.find(r);
      if (itr != _test_map.end()) {
	int v = itr->second;
	v = _ratings.rating_class(v);
	assert(v > 0);
	if (_save_ranking_file) {
	  if (_ratings.r(n, m) == .0) // skip training
	    fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, v);
	}
	
	if (j < 10) {
	  hits10++;
	  hits100++;
	} else if (j < 100) {
	  hits100++;
	}
      } else {
	if (_save_ranking_file) {
	  if (_ratings.r(n, m) == .0) // skip training
	    fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, 0);
	}
      }
    }
    mhits10 += (double)hits10 / 10;
    mhits100 += (double)hits100 / 100;
    total_users++;
  }
  if (_save_ranking_file)
    fclose(f);
  fprintf(_pf, "%.5f\t%.5f\n", 
	  (double)mhits10 / total_users, 
	  (double)mhits100 / total_users);
  fflush(_pf);
}

void
GAPRec::validation_likelihood(bool bias)
{
  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  for (CountMap::const_iterator i = _validation_map.begin();
       i != _validation_map.end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    yval_t r = i->second;
    double u = bias ? pair_likelihood_bias(n,m,r) :  pair_likelihood(n,m,r);
    s += u;
    k += 1;
  }
  info("s = %.5f\n", s);
  fprintf(_vf, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(_vf);
  double a = s / k;

  bool stop = false;
  int why = -1;
  if (_iter > 1000) {
    if (a > _prev_h && _prev_h != 0 && fabs((a - _prev_h) / _prev_h) < 0.00001) {
      stop = true;
      why = 0;
    } else if (a < _prev_h)
      _nh++;
    else if (a > _prev_h)
      _nh = 0;

    if (_nh > 3) { // be robust to small fluctuations in predictive likelihood
      why = 1;
      stop = true;
    }
  }
  _prev_h = a;
  FILE *f = fopen(Env::file_str("/max.txt").c_str(), "w");
  fprintf(f, "%d\t%d\t%.5f\t%d\n", 
	  _iter, duration(), a, why);
  fclose(f);
  if (stop) {
    do_on_stop();
    exit(0);
  }
}

void
GAPRec::do_on_stop()
{
  save_model();
  auc(_env.bias);
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
  return y * log(s) - s;
}

double
GAPRec::pair_likelihood_bias(uint32_t p, uint32_t q, yval_t y) const
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += etheta[p][k] * ebeta[q][k];
  s +=  _Eu[p] + _Ei[q];
  if (s < 1e-30)
    s = 1e-30;
  info("%d, %d, s = %f, f(y) = %ld\n", p, q, s, factorial(y));
  return y * log(s) - s;
}

double
GAPRec::pair_likelihood_binary(uint32_t p, uint32_t q, yval_t y) const
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  double s = .0;
  for (uint32_t k = 0; k < _k; ++k)
    s += etheta[p][k] * ebeta[q][k];
  if (s < 1e-30)
    s = 1e-30;
  info("%d, %d, s = %f, f(y) = %ld\n", p, q, s, factorial(y));
  return y == 0 ? -s : log(1 - exp(-s));
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
      uint32_t mov = (*movies)[m];
      assert (_ratings.r(user, mov) > 0);
      
      r.first = user;
      r.second = mov;
      break;
    }
  } while (1);
}

// uniform sample of (user, item) tuples in training
void
GAPRec::get_random_rating3(Rating &r) const
{
  const vector<Rating> &allratings = _ratings.allratings();
  uint32_t p = gsl_rng_uniform_int(_r, allratings.size());
  r = allratings[p];
}

void
GAPRec::save_model()
{
  save_item_state("/beta.txt", _Ebeta);
  save_item_state("/c.txt", _ccurr);
  save_state("/d.txt", _dcurr);
  save_state("/i.txt", _icurr);

  save_user_state("/theta.txt", _Etheta);
  save_user_state("/a.txt", _acurr);
  save_state("/b.txt", _bcurr);
  save_state("/u.txt", _ucurr);
  save_state("/Eu.txt", _Eu);
  save_state("/Ei.txt", _Ei);
}

void
GAPRec::save_user_state(string s, const Matrix &mat)
{
  FILE * tf = fopen(Env::file_str(s.c_str()).c_str(), "w");
  const double **gd = mat.data();
  for (uint32_t i = 0; i < _n; ++i) {
    const IDMap &m = _ratings.seq2user();
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) {
      fprintf(tf,"%d\t", i);
      debug("looking up i %d\n", i);
      fprintf(tf,"%d\t", (*idt).second);
      for (uint32_t k = 0; k < _k; ++k) {
	if (k == _k - 1)
	  fprintf(tf,"%.5f\n", gd[i][k]);
	else
	  fprintf(tf,"%.5f\t", gd[i][k]);
      }
    }
  }
  fclose(tf);
}

void
GAPRec::save_item_state(string s, const Matrix &mat)
{
  FILE *tf = fopen(Env::file_str(s.c_str()).c_str(), "w");
  const double **cd = mat.data();
  for (uint32_t i = 0; i < _m; ++i) {
    const IDMap &m = _ratings.seq2movie();
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) {
      fprintf(tf,"%d\t", i);
      debug("looking up i %d\n", i);
      fprintf(tf,"%d\t", (*idt).second);
      for (uint32_t k = 0; k < _k; ++k) {
	if (k == _k - 1)
	  fprintf(tf,"%.5f\n", cd[i][k]);
	else
	  fprintf(tf,"%.5f\t", cd[i][k]);
      }
    }
  }
  fclose(tf);
}


void
GAPRec::save_state(string s, const Array &mat)
{
  FILE *tf = fopen(Env::file_str(s.c_str()).c_str(), "w");
  const double *cd = mat.data();
  for (uint32_t k = 0; k < mat.size(); ++k) {
    if (k == _k - 1)
      fprintf(tf,"%.5f\n", cd[k]);
    else
      fprintf(tf,"%.5f\t", cd[k]);
  }
  fclose(tf);
}

void
GAPRec::load_beta_and_theta()
{
  load_file("theta.txt", _Etheta);
  load_file("beta.txt", _Ebeta);
  if (_env.bias) {
    load_file("Eu.txt", _Eu);
    load_file("Ei.txt", _Ei);
  }
}

void
GAPRec::load_file(string s, Array &mat)
{
  double *md = mat.data();
  FILE *f = fopen(s.c_str(), "r");
  assert (f);
  
  uint32_t k = 0;
  uint32_t K = mat.n();
  uint32_t sz = 32 * K;
  char *line = (char *)malloc(sz);
  while (!feof(f)) {
    if (fgets(line, sz, f) == NULL)
      break;
    uint32_t t = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (q == p) 
	break;
      p = q;
      md[k] = d;
      k++;
    } while (p != NULL);
    debug("read %d entries\n", k);
    memset(line, 0, sz);
  }
  assert (k == K);
  fclose(f);
  free(line);
}

void
GAPRec::load_file(string s, Matrix &mat)
{
  FILE *f = fopen(s.c_str(), "r");
  double **md = mat.data();
  assert (f);
  uint32_t n = 0;
  int sz = 32*_k;
  char *line = (char *)malloc(sz);
  while (!feof(f)) {
    if (fgets(line, sz, f) == NULL)
      break;
    //assert (fscanf(gammaf, "%[^\n]", line) > 0);
    debug("line = %s\n", line);
    uint32_t k = 0;
    char *p = line;
    do {
      char *q = NULL;
      double d = strtod(p, &q);
      if (q == p) {
	if (k < _k - 1) {
	  fprintf(stderr, "error parsing theta file\n");
	  assert(0);
	}
	break;
      }
      p = q;
      if (k >= 2) // skip node id and seq
	md[n][k-2] = d;
      k++;
    } while (p != NULL);
    n++;
    memset(line, 0, sz);
  }
  lerr("read %d lines\n", n);
  assert (n <= _n || n <= _m);
  fclose(f);
  free(line);
}

void
GAPRec::gen_ranking_for_users()
{
  load_beta_and_theta();

  char buf[4096];
  sprintf(buf, "%s/test_users.tsv", _env.datfname.c_str());
  FILE *f = fopen(buf, "r");
  assert(f);
  _ratings.read_test_users(f, &_sampled_users);
  fclose(f);
  _save_ranking_file = true;
  auc(_env.bias);
  _save_ranking_file = false;
  printf("DONE writing ranking.tsv in output directory\n");
  fflush(stdout);
}

void
GAPRec::analyze()
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  load_beta_and_theta();

  // pick a user, one with reasonable activity and is in the test set
  ValMap xmap;
  for (CountMap::const_iterator i = _test_map.begin();
       i != _test_map.end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    const vector<uint32_t> *movies = _ratings.get_movies(n);
    if (movies && movies->size() > 200 && movies->size() <= 300)
      xmap[n]++;
  }
  KVArray kvlist(xmap.size());
  uint32_t j = 0;
  for (ValMap::const_iterator i = xmap.begin();
       i != xmap.end(); ++i) {
    uint32_t n = i->first;
    uint32_t v = i->second;
    kvlist[j++] = KV(n,v);
  }
  kvlist.sort_by_value();
  uint32_t user;
  for (uint32_t i = 0; i < kvlist.size(); ++i) {
    if (kvlist[i].second > 10) {
      user = kvlist[i].first;
      break;
    }
  }

  const IDMap &seq2user = _ratings.seq2user();
  IDMap::const_iterator idt = seq2user.find(user);
  assert (idt != seq2user.end());
  printf("picked user %d (true id: %d)\n", user, idt->second);

  write_etheta(user);

  // list users top rated movies
  vector<uint32_t> liked_movies;
  bool x = true;
  const vector<uint32_t> *movies = _ratings.get_movies(user);
  for (uint32_t i = 0; i < movies->size(); ++i) {
    if (_ratings.r(user, movies->at(i)) >= 4) {
      liked_movies.push_back(movies->at(i));
      if (x) {
	write_ebeta(movies->at(i));
	x = false;
      }
    }
  }
  write_movie_list("likes", user, liked_movies);

  // xxx bias off
  // list users top recommended movies
  KVArray mlist(_m);
  vector<uint32_t> recommended_movies;
  for (uint32_t m = 0; m < _m; ++m) {
    if (_ratings.r(user,m) > 0) { // skip training
      mlist[m].first = m;
      mlist[m].second = .0;
      continue;
    }
    double u = _env.bias ? link_prob_bias(user, m) : link_prob(user,m);
    mlist[m].first = m;
    mlist[m].second = u;
  }
  uint32_t hits10 = 0, hits100 = 0;
  mlist.sort_by_value();
  for (uint32_t j = 0; j < mlist.size() && j < _topN_by_user; ++j) {
    KV &kv = mlist[j];
    uint32_t m = kv.first;
    //double pred = kv.second;
    //Rating r(n, m);
    recommended_movies.push_back(m);
  }
  write_movie_list("recommended", user, recommended_movies);

  // find top 10 factors for user
  KVArray kvlist2(_k);
  for (uint32_t i = 0; i < _k; ++i) {
    kvlist2[i].first = i;
    kvlist2[i].second = etheta[user][i];
  }
  kvlist2.sort_by_value();
  //vector<uint32_t> factors;
  //for (uint32_t j = 0; j < kvlist2.size() && j < 10; ++j) 
  //  factors.push_back(kvlist2[j].first);
  
  KVArray kvlist3(_m);
  uint32_t nf = 0;
  for (uint32_t i = 0; i < kvlist2.size() && nf < 10; ++i) {
    vector<uint32_t> top_items;
    kvlist3.zero();
    uint32_t k = kvlist2[i].first;
    printf("top factor: %d\n", k);
    // find the top 10 items of these factors
    for (uint32_t m = 0; m < _m; ++m) {
      kvlist3[m].first = m;
      kvlist3[m].second = ebeta[m][k];
    }
    kvlist3.sort_by_value();
    if (kvlist3[0].second < 0.01) // pick a strong factor
      continue;
    bool cancel = false;
    for (uint32_t j = 0; j < kvlist3.size() && j < 10; ++j)  {
      top_items.push_back(kvlist3[j].first);
      if (kvlist3[j].second < 0.001) {
	cancel = true;
	break;
      }
      printf("pushing %d,%.3f into top item\n", kvlist3[j].first, kvlist3[j].second);
      fflush(stdout);
    }
    if (cancel)
      continue;
    nf++;
    char buf[512];
    sprintf(buf, "factor_rank%d_id%d", i,k);
    write_movie_list(buf, user, top_items);
  }
}



void
GAPRec::analyze_factors()
{
  const double **etheta = _Etheta.const_data();
  const double **ebeta = _Ebeta.const_data();
  load_beta_and_theta();

  KVArray kvlist3(_m);
  for (uint32_t k = 0; k < _k; ++k) {
    vector<uint32_t> top_items;
    kvlist3.zero();
    // find the top 10 items of these factors
    for (uint32_t m = 0; m < _m; ++m) {
      kvlist3[m].first = m;
      kvlist3[m].second = ebeta[m][k];
    }
    kvlist3.sort_by_value();
    if (kvlist3[0].second < 0.01) // pick a strong factor
      continue;
    bool cancel = false;
    for (uint32_t j = 0; j < kvlist3.size() && j < 10; ++j)  {
      top_items.push_back(kvlist3[j].first);
      if (kvlist3[j].second < 0.001) {
	cancel = true;
	break;
      }
      //printf("pushing %d,%.3f into top item\n", kvlist3[j].first, kvlist3[j].second);
      //fflush(stdout);
    }
    if (cancel)
      continue;
    char buf[512];
    sprintf(buf, "factor_%d", k);
    write_movie_list(buf, top_items);
  }
}
 

void
GAPRec::write_movie_list(string label, uint32_t u, 
			 const vector<uint32_t> &movies)
{
  const IDMap &seq2user = _ratings.seq2user();
  const IDMap &seq2movie = _ratings.seq2movie();
  char s[1024];

  IDMap::const_iterator idt = seq2user.find(u);
  assert (idt != seq2user.end());
  uint32_t user = idt->second;
  sprintf(s, "/user_%s_%d.txt", label.c_str(), user);
  FILE *f = fopen(Env::file_str(s).c_str(), "w");
  for (uint32_t i = 0; i < movies.size(); ++i) {
    IDMap::const_iterator mdt = seq2movie.find(movies[i]);
    assert (mdt != seq2movie.end());
    uint32_t mov = mdt->second;
    fprintf(f, "%d\t%d\t%s\t%s\n", 
	    movies[i],
	    mov,
	    _ratings.movie_name(movies[i]).c_str(), 
	    _ratings.movie_type(movies[i]).c_str());
  }
  fflush(f);
  fclose(f);
}


void
GAPRec::write_movie_list(string label,
			 const vector<uint32_t> &movies)
{
  const IDMap &seq2user = _ratings.seq2user();
  const IDMap &seq2movie = _ratings.seq2movie();
  char s[1024];
  sprintf(s, "/%s.txt", label.c_str());
  FILE *f = fopen(Env::file_str(s).c_str(), "w");
  for (uint32_t i = 0; i < movies.size(); ++i) {
    IDMap::const_iterator mdt = seq2movie.find(movies[i]);
    assert (mdt != seq2movie.end());
    uint32_t mov = mdt->second;
    fprintf(f, "%d\t%d\t%s\t%s\n", 
	    movies[i],
	    mov,
	    _ratings.movie_name(movies[i]).c_str(), 
	    _ratings.movie_type(movies[i]).c_str());
  }
  fflush(f);
  fclose(f);
}



void
GAPRec::write_etheta(uint32_t user)
{
  const IDMap &seq2user = _ratings.seq2user();
  IDMap::const_iterator idt = seq2user.find(user);
  assert (idt != seq2user.end());
  uint32_t u = idt->second;
  char s[1024];
  sprintf(s, "/user_etheta_%d.txt", u);
  FILE *f = fopen(Env::file_str(s).c_str(), "w");
  const double **etheta = _Etheta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    fprintf(f, "%d\t%.5f\n", k, etheta[user][k]);
  fclose(f);
}

void
GAPRec::write_ebeta(uint32_t item)
{
  char s[1024];
  sprintf(s, "/movie_ebeta_%d.txt", item);
  FILE *f = fopen(Env::file_str(s).c_str(), "w");
  const double **ebeta = _Ebeta.const_data();
  for (uint32_t k = 0; k < _k; ++k)
    fprintf(f, "%d\t%d\t%.5f\n", item, k, ebeta[item][k]);
  fclose(f);  
}
