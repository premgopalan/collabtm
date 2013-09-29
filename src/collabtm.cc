#include "collabtm.hh"

CollabTM::CollabTM(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _nusers(env.nusers), 
    _ndocs(env.ndocs),
    _nvocab(env.nvocab),
    _k(env.k),
    _iter(0),
    _start_time(time(0)),
    _theta(0.3, 0.3, _ndocs,_k,&_r),
    _beta(0.3, 0.3, _nvocab,_k,&_r),
    _x(0.3, 0.3, _nusers,_k,&_r),
    _epsilon(0.3, 0.3, _ndocs,_k,&_r),
    _a(0.3, 0.3, _ndocs,&_r),
    _tau(_ndocs, _k, 2)
{
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed)
    gsl_rng_set(_r, _env.seed);

  _af = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_af)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
}

void
CollabTM::initialize()
{
  _theta.initialize();
  _beta.initialize();
  _x.initialize();
  _epsilon.initialize();
  _a.initialize();
}

void
CollabTM::get_phi(GPBase<Matrix> &a, uint32_t ai, 
		  GPBase<Matrix> &b, uint32_t bi, 
		  Array &phi)
{
  assert (phi.size() == a.k() &&
	  phi.size() == b.k());
  debug("ai = %d, bi = %d\n", ai, bi);
  assert (ai < a.n() && bi < b.n());
  const double  **eloga = _theta.expected_v().const_data();
  const double  **elogb = _beta.expected_logv().const_data();
  phi.zero();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = eloga[ai][k] + elogb[bi][k];
  phi.lognormalize();
}


void
CollabTM::get_tau(GPBase<Matrix> &a, GPBase<Matrix> &b,
		  uint32_t nd, D3 &tau)
{
  assert (tau.m() == a.n() &&
	  tau.m() == b.n());
  assert (nd < a.n());
  const double  **eloga = _theta.expected_v().const_data();
  const double  **elogb = _beta.expected_logv().const_data();
  double ***taud = _tau.data();
  Array p(2);
  for (uint32_t k = 0; k < _k; ++k) {
    p[0] = eloga[nd][k];
    p[1] = elogb[nd][k];
    p.lognormalize();
    taud[nd][k][0] = p[0];
    taud[nd][k][1] = p[1];
  }
}

void
CollabTM::batch_infer()
{
  initialize();

  Array phi(_k);
  Array tauphi(_k);
  double ***taud = _tau.data();

  while(1) {

    for (uint32_t nd = 0; nd < _ndocs; ++nd) {

      get_tau(_theta, _epsilon, nd, _tau);

      const WordVec *w = _ratings.get_words(nd);
      for (uint32_t nw = 0; w && nw < w->size(); nw++) {
	WordCount p = (*w)[nw];
	uint32_t word = p.first;
	uint32_t count = p.second;

	get_phi(_theta, nd, _beta, word, phi);
	if (count > 1)
	  phi.scale(count);

	_theta.update_shape_next(nd, phi);
	_beta.update_shape_next(word, phi);
      }
    }

    for (uint32_t nu = 0; nu < _nusers; ++nu) {

      const vector<uint32_t> *docs = _ratings.get_movies(nu);
      for (uint32_t j = 0; j < docs->size(); ++j) {
	uint32_t nd = (*docs)[j];
	yval_t y = _ratings.r(nu,nd);

	get_phi(_x, nu, _theta, nd, phi);
	if (y > 1)
	  phi.scale(y);

	// todo: optimize
	// rather than multiply each time, add all the phi
	// and scale by tau at the end;
	// then do theta.update_shape_next()

	for (uint32_t k = 0; k < _k; ++k) 
	  tauphi0[k] = phi[k] * taud[nd][k][0];

	_theta.update_shape_next(nd, phi);
	_x.update_shape_next(nu, phi);
	_a.update_shape_next(nd, y); // since \sum_k \phi_k = 1

	get_phi(_x, nu, _epsilon, nd, phi);
	if (y > 1)
	  phi.scale(y);

	_epsilon.update_shape_next(nd, phi);
	_x.update_shape_next(nu, phi);
      }
    }

    update_all_rates();
    swap_all();
    compute_all_expectations();

    if (_iter % 10 == 0) {
      printf("Iteration %d\n", _iter);
      fflush(stdout);
      approx_log_likelihood();
    }

    _iter++;
  }
}


void
CollabTM::update_all_rates()
{
  // update theta rate
  Array xsum(_k);
  _x.sum_rows(xsum);
  Array betasum(_k);
  _beta.sum_rows(betasum);
  _theta.update_rate_next(betasum);
  _theta.update_rate_next(xsum, _a.expected_v());
    
  // update beta rate
  Array thetasum(_k);
  _theta.sum_rows(thetasum);
  _beta.update_rate_next(thetasum);

  // update x rate
  Array scaledthetasum(_k);
  _theta.scaled_sum_rows(scaledthetasum, _a.expected_v());
  Array epsilonsum(_k);
  _epsilon.sum_rows(epsilonsum);
  _x.update_rate_next(scaledthetasum);
  _x.update_rate_next(epsilonsum);

  // update epsilon rate
  _epsilon.update_rate_next(xsum);

  // update 'a' rate
  Array arate(_ndocs);
  Matrix &theta_ev = _theta.expected_v();
  const double **theta_evd = theta_ev.const_data();
  for (uint32_t nd = 0; nd < _ndocs; ++nd)
    for (uint32_t k = 0; k < _k; ++k)
      arate[nd] += xsum[k] * theta_evd[nd][k];
  _a.update_rate_next(arate);
}

void
CollabTM::swap_all()
{
  _theta.swap();
  _beta.swap();
  _epsilon.swap();
  _a.swap();
  _x.swap();
}

void
CollabTM::compute_all_expectations()
{
  _theta.compute_expectations();
  _beta.compute_expectations();
  _epsilon.compute_expectations();
  _a.compute_expectations();
  _x.compute_expectations();  
}

void
CollabTM::approx_log_likelihood()
{
  const double ** etheta = _theta.expected_v().const_data();
  const double ** elogtheta = _theta.expected_logv().const_data();
  const double ** ebeta = _beta.expected_v().const_data();
  const double ** elogbeta = _beta.expected_logv().const_data();

  const double ** ex = _x.expected_v().const_data();
  const double ** elogx = _x.expected_logv().const_data();
  const double ** eepsilon = _epsilon.expected_v().const_data();
  const double ** elogepsilon = _epsilon.expected_logv().const_data();
  const double *ea = _a.expected_v().const_data();
  const double *eloga = _a.expected_logv().const_data();

  double s = .0;

  Array phi(_k);
  for (uint32_t nd = 0; nd < _ndocs; ++nd) {
    const WordVec *w = _ratings.get_words(nd);
    for (uint32_t nw = 0; w && nw < w->size(); nw++) {
      WordCount p = (*w)[nw];
      uint32_t word = p.first;
      uint32_t count = p.second;

      get_phi(_theta, nd, _beta, word, phi);
      
      double v = .0;
      for (uint32_t k = 0; k < _k; ++k) 
	v += count * phi[k] * (elogtheta[nd][k] + elogbeta[word][k] - log(phi[k]));
      s += v;
      
      for (uint32_t k = 0; k < _k; ++k)
	s -= etheta[nd][k] * ebeta[word][k];
    }
  }

  for (uint32_t nu = 0; nu < _nusers; ++nu) {

    const vector<uint32_t> *docs = _ratings.get_movies(nu);
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      yval_t y = _ratings.r(nu,nd);

      get_phi(_x, nu, _theta, nd, phi);

      double v = .0;
      for (uint32_t k = 0; k < _k; ++k) 
	v += y * phi[k] * (eloga[nd] + elogx[nu][k] + elogtheta[nd][k] - log(phi[k]));
      s += v;
      
      for (uint32_t k = 0; k < _k; ++k)
	s -= ex[nu][k] * etheta[nd][k] * ea[nd];

      get_phi(_x, nu, _theta, nd, phi);

      v = .0;
      for (uint32_t k = 0; k < _k; ++k) 
	v += y * phi[k] * (elogx[nu][k] + elogepsilon[nd][k] - log(phi[k]));
      s += v;

      for (uint32_t k = 0; k < _k; ++k)
	s -= eepsilon[nd][k] * ex[nu][k];
    }
  }

  s += _theta.compute_elbo_term();
  s += _beta.compute_elbo_term();
  s += _x.compute_elbo_term();
  s += _epsilon.compute_elbo_term();
  s += _a.compute_elbo_term();

  fprintf(_af, "%.5f\n", s);
  fflush(_af);
}
