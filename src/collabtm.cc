#include "collabtm.hh"

CollabTM::CollabTM(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _nusers(env.nusers), 
    _ndocs(env.ndocs),
    _nvocab(env.nvocab),
    _k(env.k),
    _iter(0),
    _start_time(time(0)),
    _theta("theta", 0.3, 0.3, _ndocs,_k,&_r),
    _beta("beta", 0.3, 0.3, _nvocab,_k,&_r),
    _x("x", 0.3, 0.3, _nusers,_k,&_r),
    _epsilon("epsilon", 0.3, 0.3, _ndocs,_k,&_r),
    _a("a", 0.3, 0.3, _ndocs,&_r),
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
  assert (ai < a.n() && bi < b.n());
  const double  **eloga = a.expected_v().const_data();
  const double  **elogb = b.expected_logv().const_data();
  phi.zero();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = eloga[ai][k] + elogb[bi][k];
  phi.lognormalize();
}


void
CollabTM::get_xi(uint32_t user, uint32_t doc, Array &phi)
{
  const double  **elogx = _x.expected_v().const_data();
  const double  **elogtheta = _theta.expected_logv().const_data();
  const double  **elogepsilon = _epsilon.expected_logv().const_data();
  double ***taud = _tau.data();
  phi.zero();
  for (uint32_t k = 0; k < _k; ++k) {
    phi[k] = elogx[user][k] + taud[doc][k][0] * elogtheta[doc][k] + \
      taud[doc][k][1] * elogepsilon[user][k];
    phi[k] -= taud[doc][k][0] * log (taud[doc][k][0]) + \
      taud[doc][k][1] * log (taud[doc][k][1]);
  }
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
  const double  **elogb = _epsilon.expected_logv().const_data();
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
  compute_all_expectations();
  approx_log_likelihood();

  Array phi(_k);
  Array xi(_k);
  Array tauphi0(_k), tauphi1(_k);
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

	get_xi(nu, nd, xi);
	if (y > 1)
	  xi.scale(y);

	// todo: optimize
	// rather than multiply each time, add all the xi
	// and scale by tau at the end;
	// then do theta.update_shape_next()

	for (uint32_t k = 0; k < _k; ++k)  {
	  tauphi0[k] = xi[k] * taud[nd][k][0];
	  tauphi1[k] = xi[k] * taud[nd][k][1];
	}

	_theta.update_shape_next(nd, tauphi0);
	_epsilon.update_shape_next(nd, tauphi1);
	_x.update_shape_next(nu, xi);
	_a.update_shape_next(nd, y); // since \sum_k \phi_k = 1
      }
    }

    update_all_rates();
    swap_all();
    compute_all_expectations();
    
    if (_iter % 10 == 0) {
      printf("Iteration %d\n", _iter);
      fflush(stdout);
      approx_log_likelihood();
      save_model();
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
  double ***taud = _tau.data();
  Array phi(_k);
  Array xi(_k);
  for (uint32_t nd = 0; nd < _ndocs; ++nd) {
    get_tau(_theta, _epsilon, nd, _tau);

    const WordVec *w = _ratings.get_words(nd);
    for (uint32_t nw = 0; w && nw < w->size(); nw++) {
      WordCount p = (*w)[nw];
      uint32_t word = p.first;
      uint32_t count = p.second;
      
      get_phi(_theta, nd, _beta, word, phi);
      
      double v = .0;
      for (uint32_t k = 0; k < _k; ++k) 
	v += count * phi[k] * (elogtheta[nd][k] +			\
			       elogbeta[word][k] - log(phi[k]));
      s += v;
      
      for (uint32_t k = 0; k < _k; ++k)
	s -= etheta[nd][k] * ebeta[word][k];
    }
  }

  lerr("E1: s = %f\n", s);
  
  for (uint32_t nu = 0; nu < _nusers; ++nu) {
    
    const vector<uint32_t> *docs = _ratings.get_movies(nu);
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      yval_t y = _ratings.r(nu,nd);
      
      get_xi(nu, nd, xi);
      
      double v = .0;
      for (uint32_t k = 0; k < _k; ++k) {
	double t = taud[nd][k][0] * log (taud[nd][k][0])	\
	  + taud[nd][k][1] * log (taud[nd][k][1]);
	v += y * xi[k] * (eloga[nd] + elogx[nu][k]		\
			  + taud[nd][k][0] * elogtheta[nd][k]	\
			  + taud[nd][k][1] * elogepsilon[nu][k] \
			  - t - log(xi[k]));
      }
      s += v;
      
      for (uint32_t k = 0; k < _k; ++k)
	s -= ex[nu][k] * (etheta[nd][k] + eepsilon[nd][k]) * ea[nd];
    }
  }

  lerr("E2: s = %f\n", s);

  s += _theta.compute_elbo_term();
  s += _beta.compute_elbo_term();
  s += _x.compute_elbo_term();
  s += _epsilon.compute_elbo_term();
  s += _a.compute_elbo_term();

  lerr("E3: s = %f\n", s);

  fprintf(_af, "%.5f\n", s);
  fflush(_af);
}

void
CollabTM::save_model()
{
  IDMap idmap; // null map
  _x.save_state(_ratings.seq2user());
  _theta.save_state(_ratings.seq2movie());
  _epsilon.save_state(_ratings.seq2movie());
  _beta.save_state(idmap);
  _a.save_state(_ratings.seq2movie());
}


void
CollabTM::save_state(string s, const Array &mat)
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

