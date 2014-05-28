#include "collabtm.hh"

CollabTM::CollabTM(Env &env, Ratings &ratings)
  : _env(env), _ratings(ratings),
    _nusers(env.nusers), 
    _ndocs(env.ndocs),
    _nvocab(env.nvocab),
    _k(env.k),
    _rhou(_nusers),
    _rhow(_nvocab),
    _tau0(1024), _kappa(.5),
    _userc(_nusers),
    _wordc(_nvocab),
    _iter(0),
    _start_time(time(0)),
    _theta("theta", 0.3, 0.3, _ndocs,_k,&_r),
    _beta("beta", 0.3, 0.3, _nvocab,_k,&_r),
    _x("x", 0.3, 0.3, _nusers,_k,&_r),
    _epsilon("epsilon", 0.3, 0.3, _ndocs,_k,&_r),
    _a("a", 0.3, 0.3, _ndocs, &_r),
    _cstheta("cstheta", 0.3, 0.3, _ndocs,_k,&_r),
    _prev_h(.0), _nh(0),
    _save_ranking_file(false),
    _topN_by_user(100),
    _ncsdoc_seq(0)
{
  gsl_rng_env_setup();
  const gsl_rng_type *T = gsl_rng_default;
  _r = gsl_rng_alloc(T);
  if (_env.seed)
    gsl_rng_set(_r, _env.seed);

  _userc.zero(); 

  _af = fopen(Env::file_str("/logl.txt").c_str(), "w");
  if (!_af)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  _vf = fopen(Env::file_str("/validation.txt").c_str(), "w");
  if (!_vf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _cs_tf = fopen(Env::file_str("/coldstart_test.txt").c_str(), "w");
  if (!_cs_tf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _cs_tf2 = fopen(Env::file_str("/coldstart_test_perdoc.txt").c_str(), "w");
  if (!_cs_tf2)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _tf = fopen(Env::file_str("/test.txt").c_str(), "w");
  if (!_tf)  {
    printf("cannot open heldout file:%s\n",  strerror(errno));
    exit(-1);
  }
  _pf = fopen(Env::file_str("/precision.txt").c_str(), "w");
  if (!_pf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  _cs_pf = fopen(Env::file_str("/coldstart_precision.txt").c_str(), "w");
  if (!_cs_pf)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  _df = fopen(Env::file_str("/ndcg.txt").c_str(), "w");
  if (!_df)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  _cs_df = fopen(Env::file_str("/coldstart_ndcg.txt").c_str(), "w");
  if (!_cs_df)  {
    printf("cannot open logl file:%s\n",  strerror(errno));
    exit(-1);
  }
  load_validation_and_test_sets();
}

void
CollabTM::initialize()
{
  if (_env.use_docs) {
    if (_env.lda) { // fix lda topics and doc memberships

      // beta and theta are fixed after loading
      assert (_env.fixed_doc_param);

      _beta.set_to_prior_curr();
      _beta.set_to_prior();

      _theta.set_to_prior_curr();
      _theta.set_to_prior();

      _beta.load_from_lda(_env.datfname, 0.01, _k); // eek! fixme.
      _theta.load_from_lda(_env.datfname, 0.1, _k);
      lerr("loaded lda fits");
      
    } else if (_env.lda_init) { // lda based init
      
      _beta.set_to_prior_curr();
      _beta.set_to_prior();

      _theta.set_to_prior_curr();
      _theta.set_to_prior();
      
      _beta.load_from_lda(_env.datfname, 0.01, _k); // eek! fixme.
      _theta.load_from_lda(_env.datfname, 0.1, _k);

    } else { // random init
      
      _beta.initialize();
      _theta.initialize();
      
      _theta.initialize_exp();
      _beta.initialize_exp();
    }
  }

  if (_env.use_ratings) {

    if (_env.use_docs) {
      _x.set_to_prior();
      _x.set_to_prior_curr();
      
      _epsilon.set_to_prior();
      _epsilon.set_to_prior_curr();

      _x.compute_expectations();
      _epsilon.compute_expectations();

      if (!_env.fixeda) {
	_a.set_to_prior();
	_a.set_to_prior_curr();
	_a.compute_expectations();
      }
    } else {
      _x.initialize_exp();
      _epsilon.initialize_exp();
      
      if (!_env.fixeda) {
	_a.initialize();
	_a.compute_expectations();
      }
    }
    
  }
}

void
CollabTM::initialize_perturb_betas()
{
  if (_env.use_docs) {
    _beta.initialize();
    _theta.set_to_prior_curr();
    _theta.set_to_prior();

    _theta.compute_expectations();
    _beta.compute_expectations();
  }

  if (_env.use_ratings) {
    _x.set_to_prior_curr();
    _x.set_to_prior();
    _epsilon.set_to_prior_curr();
    _epsilon.set_to_prior();
    _x.compute_expectations();
    _epsilon.compute_expectations();

    if (!_env.fixeda) {
      _a.set_to_prior_curr();
      _a.set_to_prior();
      _a.compute_expectations();
    }
  }
}

void
CollabTM::seq_init_helper()
{
  if (!_env.fixed_doc_param) {
    _beta.set_to_prior_curr();
    _beta.set_to_prior();
    _beta.compute_expectations();
    
    _theta.set_to_prior_curr();
    _theta.set_to_prior();
    _theta.compute_expectations();

  } else { // load beta and theta from saved state

    _beta.set_to_prior();
    _theta.set_to_prior();
    
    _beta.load();
    _theta.load();
  }

  if (_env.use_ratings) {
    _x.set_to_prior_curr();
    _x.set_to_prior();
    _x.compute_expectations();

    _epsilon.set_to_prior_curr();
    _epsilon.set_to_prior();
    _epsilon.compute_expectations();
    
    if (!_env.fixeda) {
      _a.set_to_prior();
      _a.set_to_prior_curr();
      _a.compute_expectations();
    }
  }
}

void
CollabTM::load_validation_and_test_sets()
{
  char buf[4096];
  sprintf(buf, "%s/validation.tsv", _env.datfname.c_str());
  FILE *validf = fopen(buf, "r");
  assert(validf);
  if (_env.dataset == Env::NYT)
    _ratings.read_nyt_train(validf, &_validation_map);
  else
    _ratings.read_generic(validf, &_validation_map);
  fclose(validf);

  sprintf(buf, "%s/test.tsv", _env.datfname.c_str());
  FILE *testf = fopen(buf, "r");
  assert(testf);
  if (_env.dataset == Env::NYT)
    _ratings.read_nyt_train(validf, &_test_map);
  else
    _ratings.read_generic(testf, &_test_map);
  fclose(testf);
  printf("+ loaded validation and test sets from %s\n", _env.datfname.c_str());
  fflush(stdout);

  // select some documents for cold start recommendations
  // remove them from training, test and validation
  Env::plog("test ratings before removing heldout cold start docs", _test_map.size());
  Env::plog("validation ratings before removing heldout cold start docs", _validation_map.size());
  Env::plog("cold start docs", _env.heldout_items_ratio * _env.ndocs); 

  uint32_t c = 0;
  while (c < _env.heldout_items_ratio * _env.ndocs) {
    uint32_t n = gsl_rng_uniform_int(_r, _env.ndocs);
    const vector<uint32_t> *u = _ratings.get_users(n);
    assert(u);
    uint32_t nusers = u->size();
    if (nusers < 10)
      continue;
    _cold_start_docs[n] = true;
    c++;
  }
  Env::plog("number of heldout cold start docs", _cold_start_docs.size());

  // create sequence ids for cold start docs
  _ncsdoc_seq = 0;
  for (MovieMap::const_iterator i = _cold_start_docs.begin(); 
       i != _cold_start_docs.end(); ++i) {
    _doc_to_cs_idmap[i->first] = _ncsdoc_seq++;
  }

  CountMap::iterator i = _test_map.begin(); 
  while (i != _test_map.end()) {
    const Rating &r = i->first;
    MovieMap::const_iterator itr = _cold_start_docs.find(r.second);
    if (itr != _cold_start_docs.end()) {
      // insert into coldstart test map
      _coldstart_test_map[r] = i->second;
      debug("test: erasing rating r (%d,%d) in heldout cold start", 
	    _ratings.to_user_id(r.first), 
	    _ratings.to_movie_id(r.second));
      _test_map.erase(i++);
    } else
      ++i;
  }
  
  CountMap::iterator j = _validation_map.begin(); 
  while (j != _validation_map.end()) {
    const Rating &r = j->first;
    MovieMap::const_iterator itr = _cold_start_docs.find(r.second);
    if (itr != _cold_start_docs.end()) {
      debug("validation: erasing rating r (%d,%d) in heldout cold start", 
	    _ratings.to_user_id(r.first), 
	    _ratings.to_movie_id(r.second));
      _validation_map.erase(j++);
    } else
      ++j;
  }
  Env::plog("test ratings after", _test_map.size());
  Env::plog("validation ratings after", _validation_map.size());

  FILE *g = fopen(Env::file_str("/coldstart_docs.tsv").c_str(), "w");
  write_coldstart_docs(g, _cold_start_docs);
  fclose(g);

  // users with at least 1 coldstart test items
  for (uint32_t nu = 0; nu < _nusers; ++nu) {
    const vector<uint32_t> *docs = _ratings.get_movies(nu);

    uint32_t x = 0;
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      MovieMap::const_iterator mp = _cold_start_docs.find(nd);
      if (mp != _cold_start_docs.end()) {
	_cs_users.push_back(nu);
	break;
      }
    }
  }

  _cs_test_users = new uArray(_cs_users.size());
  for (uint32_t i = 0; i < _cs_users.size(); ++i)
    (*_cs_test_users)[i] = _cs_users[i];
  gsl_ran_shuffle(_r, (void *)&((*_cs_test_users)[0]), _cs_users.size(), sizeof(uint32_t));

  g = fopen(Env::file_str("/coldstart_test_users.tsv").c_str(), "w");
  uint32_t cc = 0;
  for (uint32_t i = 0; i < _cs_users.size() && i < 10000; ++i, cc++)
    fprintf(g, "%d\n", _ratings.to_user_id((*_cs_test_users)[i]));
  fflush(g);
  fclose(g);

  g = fopen(Env::file_str("/coldstart_test.tsv").c_str(), "w");
  for (CountMap::const_iterator itr = _coldstart_test_map.begin(); 
       itr != _coldstart_test_map.end(); ++itr) {
    const Rating &r = itr->first;
    yval_t v = itr->second;
    fprintf(g, "%d\t%d\t%d\n", r.first, r.second, v);
  }
  fclose(g);

  Env::plog("cs test ratings size", _coldstart_test_map.size());
  Env::plog("cs total users size", _cs_users.size());
  Env::plog("cs test users size", cc);
}

void
CollabTM::write_coldstart_docs(FILE *f, MovieMap &mp)
{
  for (MovieMap::const_iterator i = mp.begin(); i != mp.end(); ++i) {
    uint32_t p = i->first;
    fprintf(f, "%d\t%d\n", p, _ratings.to_movie_id(p));
  }
  fflush(f);
}

void 
CollabTM::gen_ranking_for_users()
{
    // load rating prediction parameters
    printf("loading theta: "); fflush(stdout);
    _theta.load();
    printf("done\n");
    printf("loading epsilon: "); fflush(stdout);
    _epsilon.load();
    printf("done\n"); 
    printf("loading x: "); fflush(stdout);
    _x.load();
    printf("done\n"); 

    // load test users 
    char buf[4096];
    sprintf(buf, "%s/test_users.tsv", _env.datfname.c_str());
    FILE *f = fopen(buf, "r");
    assert(f);
    _ratings.read_test_users(f, &_sampled_users);
    fclose(f);
 
    // get and output rankings
    _save_ranking_file = true;
    printf("precision\n"); fflush(stdout);
    precision();
    printf("done\n");
    printf("coldstart local inference and HOL\n"); fflush(stdout);
    coldstart_rating_likelihood();
    printf("done\n");
    printf("coldstart precision\n"); fflush(stdout);
    coldstart_precision();
    printf("done\n");
    printf("DONE writing ranking.tsv in output directory\n");

    // get and output likelihood
    bool validation=false;
    bool experiment=true;
    if(!compute_likelihood(validation)) {
        save_model();
        exit(0);
    }
}

void
CollabTM::write_mult_format()
{
  FreqMap userID_to_chong_userID; 

  // out-matrix: test, users
  FILE * outf = fopen(Env::file_str("/out-test.dat").c_str(), "w");
  for (uint32_t nu = 0; nu < _nusers; ++nu) {
    const vector<uint32_t> *docs = _ratings.get_movies(nu);

	uint32_t s = userID_to_chong_userID.size();
	userID_to_chong_userID[_ratings.to_user_id(nu)] = s;

//    uint32_t x = 0;
//    for (uint32_t j = 0; j < docs->size(); ++j) {
//      uint32_t nd = (*docs)[j];
//      MovieMap::const_iterator mp = _cold_start_docs.find(nd);
//      if (mp == _cold_start_docs.end())
//	continue;
//      x++;
//    }
//    if (x == 0)
//      continue;   // no docs in cold start for this user; skip

    //fprintf(outf, "%d\t%d", _ratings.to_user_id(nu), x);
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      
      MovieMap::const_iterator mp = _cold_start_docs.find(nd);
      if (mp == _cold_start_docs.end())
	continue;

      fprintf(outf, "%d\t%d\t1\n", _ratings.to_user_id(nu), _ratings.to_movie_id(nd));
    }
  }
  fclose(outf);

  // in-matrix & out-matrix: train, items
  outf = fopen(Env::file_str("/in-train-items.dat").c_str(), "w");
  FILE *outf2 = fopen(Env::file_str("/out-train-items.dat").c_str(), "w");
  FreqMap docID_to_chong_docID; // mapping of external IDs to dat-format IDs
  for (uint32_t nd = 0; nd < _ndocs; ++nd) {
    MovieMap::const_iterator mp = _cold_start_docs.find(nd);
    if (mp != _cold_start_docs.end()) { 
	  fprintf(outf2, "%d\t0\n", _ratings.to_movie_id(nd)); 
      continue; // a cold start doc, skip
    } 

    const vector<uint32_t> *users = _ratings.get_users(nd);
    if (users->size() == 0)
      continue;

	uint32_t s = docID_to_chong_docID.size();
	docID_to_chong_docID[_ratings.to_movie_id(nd)] = s;
    
    fprintf(outf, "%d\t%d", _ratings.to_movie_id(nd), users->size());
    fprintf(outf2, "%d\t%d", _ratings.to_movie_id(nd), users->size());
    for (uint32_t j = 0; j < users->size(); ++j) {
      uint32_t nu = (*users)[j];
      yval_t y = _ratings.r(nu,nd);
      fprintf(outf, "\t%d", userID_to_chong_userID[_ratings.to_user_id(nu)]);
      fprintf(outf2, "\t%d", userID_to_chong_userID[_ratings.to_user_id(nu)]);
    }
    fprintf(outf, "\n");
    fprintf(outf2, "\n");
  }
  fclose(outf);
		     
  // in-matrix & out-matrix: train, users
  outf = fopen(Env::file_str("/in-train-users.dat").c_str(), "w");
  outf2 = fopen(Env::file_str("/out-train-users.dat").c_str(), "w");
  for (uint32_t nu = 0; nu < _nusers; ++nu) {
    const vector<uint32_t> *docs = _ratings.get_movies(nu);
    uint32_t x = 0;
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      MovieMap::const_iterator mp = _cold_start_docs.find(nd);
      if (mp != _cold_start_docs.end()) { 
	continue;
      }
      x++;
    }
    if (x == 0)
      continue;   // all docs in cold start for this user; skip

    fprintf(outf, "%d\t%d", _ratings.to_user_id(nu), x);
    fprintf(outf2, "%d\t%d", _ratings.to_user_id(nu), x);
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      MovieMap::const_iterator mp = _cold_start_docs.find(nd);
      if (mp != _cold_start_docs.end())
	continue;
      //fprintf(outf, "\t%d:%d", _ratings.to_movie_id(nd), _ratings.r(nu,nd));
      //fprintf(outf2, "\t%d:%d", _ratings.to_movie_id(nd), _ratings.r(nu,nd));
      fprintf(outf, "\t%d", docID_to_chong_docID[_ratings.to_movie_id(nd)]);
      fprintf(outf2, "\t%d", docID_to_chong_docID[_ratings.to_movie_id(nd)]);
    }
    fprintf(outf, "\n");
    fprintf(outf2, "\n");
  }
  fclose(outf); 
  fclose(outf2); 


  MapVec test_users;
  MapVec test_docs; 
  for(CountMap::const_iterator cm = _test_map.begin(); cm != _test_map.end(); ++cm) { 
  	const Rating &r = cm->first;
	uint32_t uid = r.first;
	uint32_t mid = r.second;
	test_users[uid].push_back(mid); 
	test_docs[mid].push_back(uid); 
  }

  outf = fopen(Env::file_str("/in-test.dat").c_str(), "w");
  for (uint32_t nu = 0; nu < _nusers; ++nu) {
	if (test_users.find(nu) != test_users.end())  {
		//fprintf(outf, "%d\t%d", _ratings.to_user_id(nu), test_users[nu].size()); 
		for (uint32_t j = 0; j < test_users[nu].size(); ++j) {
			uint32_t docid = test_users[nu][j];
			MovieMap::const_iterator mp = _cold_start_docs.find(docid);
			if (mp != _cold_start_docs.end())
				continue;
		    fprintf(outf, "%d\t%d\t1\n", _ratings.to_user_id(nu), _ratings.to_movie_id(docid)); 
		}
	}
  }
  fclose(outf); 
}

void
CollabTM::get_phi(GPBase<Matrix> &a, uint32_t ai, 
		  GPBase<Matrix> &b, uint32_t bi, 
		  Array &phi)
{
  assert (phi.size() == a.k() &&
	  phi.size() == b.k());
  assert (ai < a.n() && bi < b.n());
  const double  **eloga = a.expected_logv().const_data();
  const double  **elogb = b.expected_logv().const_data();
  phi.zero();
  for (uint32_t k = 0; k < _k; ++k)
    phi[k] = eloga[ai][k] + elogb[bi][k];
  phi.lognormalize();
}


void
CollabTM::get_xi(uint32_t nu, uint32_t nd, 
		 Array &xi,
		 Array &xi_a,
		 Array &xi_b)
{
  assert (xi.size() == 2 *_k && xi_a.size() == _k && xi_b.size() == _k);
  const double  **elogx = _x.expected_logv().const_data();
  const double  **elogtheta = _theta.expected_logv().const_data();
  const double  **elogepsilon = _epsilon.expected_logv().const_data();
  xi.zero();
  for (uint32_t k = 0; k < 2*_k; ++k) {
    if (k < _k)
      xi[k] = elogx[nu][k] + elogtheta[nd][k];
    else {
      uint32_t t = k - _k;
      xi[k] = elogx[nu][t] + elogepsilon[nd][t];
    }
  }
  xi.lognormalize();
  for (uint32_t k = 0; k < 2*_k; ++k) 
    if (k < _k)
      xi_a[k] = xi[k];
    else
      xi_b[k-_k] = xi[k];
}

void
CollabTM::seq_init()
{
  seq_init_helper();

  Array thetasum(_k);
  _theta.sum_rows(thetasum);

  Array betasum(_k);
  _beta.sum_rows(betasum);

  Array phi(_k);
  uArray s(_k);
  
  for (uint32_t nd = 0; nd < _ndocs; ++nd) {

    MovieMap::const_iterator mp = _cold_start_docs.find(nd);
    if (mp != _cold_start_docs.end() && mp->second == true)
      continue;
    
    const WordVec *w = _ratings.get_words(nd);
    for (uint32_t nw = 0; w && nw < w->size(); nw++) {
      WordCount p = (*w)[nw];
      uint32_t word = p.first;
      uint32_t count = p.second;
      
      get_phi(_theta, nd, _beta, word, phi);
      
      if (_iter == 0 || _env.seq_init_samples) {
	gsl_ran_multinomial(_r, _k, count, phi.const_data(), s.data());
	_beta.update_shape_curr(word, s);
	_theta.update_shape_curr(nd, s);
      }
    }
    
    if (_iter == 0) {
      if (nd % 100 == 0) {
	_beta.update_rate_curr(thetasum);
	_beta.compute_expectations();
	_theta.update_rate_curr(betasum);
	_theta.compute_expectations();
	lerr("done document %d", nd);
      }
    }
  }
  lerr("initialization complete");
  lerr("starting batch inference");
}

void
CollabTM::batch_infer()
{
  if (!_env.seq_init && !_env.seq_init_samples) {
    if (_env.perturb_only_beta_shape)
      initialize_perturb_betas();
    else
      initialize();
  } else {
    assert (_env.use_docs);
    seq_init();
  }
  
  lerr("within batch infer");

  approx_log_likelihood();

  Array phi(_k);
  Array xi(2*_k);
  Array xi_a(_k);
  Array xi_b(_k);
  uArray s(_k);
	    
  while (1) {
    if (_env.phased)
      if (_iter < 200) {
	_env.fixed_doc_param = false;
	_env.use_docs = true;
	_env.use_ratings = false;
      } else if (_iter > 200) {
	_env.fixed_doc_param = true;
	_env.use_docs = true;
	_env.use_ratings = true;
      }
    
    if (_env.use_docs && !_env.fixed_doc_param) {

      uint32_t processed_docs = 0;
      for (uint32_t nd = 0; nd < _ndocs; ++nd) {

	MovieMap::const_iterator mp = _cold_start_docs.find(nd);
	if (mp != _cold_start_docs.end() && mp->second == true)
	  continue;

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

	if (_iter == 0)
	  processed_docs++;
      }
      if (_iter == 0)
	Env::plog("processed %d in-matrix docs in 1 iteration", processed_docs);
    }
    
    if (_env.use_ratings) {
      uint32_t processed_ratings = 0;
      for (uint32_t nu = 0; nu < _nusers; ++nu) {

	const vector<uint32_t> *docs = _ratings.get_movies(nu);
	for (uint32_t j = 0; j < docs->size(); ++j) {
	  uint32_t nd = (*docs)[j];

	  MovieMap::const_iterator mp = _cold_start_docs.find(nd);
	  if (mp != _cold_start_docs.end() && mp->second == true)
	    continue;

	  //
	  // NOTE: we don't need to check this because test, validation ratings
	  // are not included the lists maintained in the Ratings class
	  // *and* since we don't iterate over zero ratings
	  //
	  //Rating r(nu,nd);
	  //if (!is_training(r))
	  //continue;

	  yval_t y = _ratings.r(nu,nd);
	  processed_ratings++;
	  
	  assert (y > 0);
	  
	  if (_env.use_docs) {
	    get_xi(nu, nd, xi, xi_a, xi_b);
	    
	    if (y > 1) {
	      xi_a.scale(y);
	      xi_b.scale(y);
	      xi.scale(y);
	    }
	    
	    if (!_env.fixed_doc_param)
	      _theta.update_shape_next(nd, xi_a);
	    
	    _epsilon.update_shape_next(nd, xi_b);
	    _x.update_shape_next(nu, xi_a);
	    _x.update_shape_next(nu, xi_b);
	    
	    if (!_env.fixeda)
	      _a.update_shape_next(nd, y); // since \sum_k \xi_k = 1

	  } else { // ratings only

	    get_phi(_x, nu, _epsilon, nd, phi);
	    if (y > 1)
	      phi.scale(y);
	    
	    _x.update_shape_next(nu, phi);
	    _epsilon.update_shape_next(nd, phi);
	  }
	}
      }
      if (_iter == 0)
	Env::plog("processed %d in-matrix ratings in 1 iteration", processed_ratings);
    }

    if (_env.vb || (_env.vbinit && _iter < _env.vbinit_iter))
      update_all_rates_in_seq();
    else {
      update_all_rates();
      swap_all();
      compute_all_expectations();
    }

    if (_iter % 10 == 0) {
      lerr("Iteration %d\n", _iter);
      approx_log_likelihood();
      if (_env.use_ratings) {
	compute_likelihood(true);
	compute_likelihood(false);
      }
      save_model();
      if (_env.use_ratings) {
	coldstart_rating_likelihood();
	coldstart_precision();
	precision();
      }
    }
    
    if (_env.save_state_now)
      exit(0);

    _iter++;
  }
}

void 
CollabTM::online_infer()
{
    //     - We sample items. 
    //         - theta, epsilon, a all get completely updated each time. 
    //         - x and beta do not 
    //         - this simplifies book keeping around the learning rate (vs. sampling users)

    if (!_env.seq_init && !_env.seq_init_samples) {
        if (_env.perturb_only_beta_shape) {
            printf("not implemented\n"); 
            assert(false); 
            initialize_perturb_betas();
            printf("init perturb betas\n"); 
        } else { 
            printf("init normal\n"); 
            initialize();
        }
    } else {
        printf("sequential init\n"); 
        assert (_env.use_docs);
        seq_init();
    }

    approx_log_likelihood();

    if(_env.mini_batch_size > _env.ndocs) { 
        _env.mini_batch_size = _env.ndocs; // TODO: do we deal correctly larger batch sizes? 
        printf("using mini-batch sizes of %d\n", _env.mini_batch_size); 
    } 

    Array phi(_k);
    Array xi(2*_k);
    Array xi_a(_k);
    Array xi_b(_k);
    UserMap sampled_words; 

    //double sbefore, safter; 

    //_x.set_to_zero_curr(); 
    bool sample_all = false; 

    while (1) {

        if(_env.use_ratings) 
            _x.set_to_zero(); 
        if(_env.use_docs) 
            _beta.set_to_zero(); 

        _sampled_users.clear();
        _sampled_movies.clear();
        sampled_words.clear(); 
        do {
            uint32_t m = gsl_rng_uniform_int(_r, _ndocs);

            MovieMap::const_iterator itr = _sampled_movies.find(m);
            if (itr == _sampled_movies.end()) {
                _sampled_movies[m] = true;
                //printf("sampled movie %d\n", m); 
                //_ratings.seq2movie();
                IDMap::const_iterator idt = _ratings.seq2movie().find(m);
                #if 0 
                if (idt != _ratings.seq2movie().end()) 
                  fprintf(_sm, "%d\n", idt->second); 
                else
                  fprintf(_sm, "unknown\n");
                #endif 

                const vector<uint32_t> *users = _ratings.get_users(m);
                for (uint32_t i = 0; i < users->size(); ++i) {
                    uint32_t n = (*users)[i];
                    _sampled_users[n] = true;
                }
            }
        } while ( (!sample_all && _sampled_movies.size() < _env.mini_batch_size) || (sample_all && _sampled_movies.size() < _ndocs) );
        lerr("sampled %d users and %d movies", _sampled_users.size(), _sampled_movies.size());
        if (sample_all)
            printf("sampled %d\n", _sampled_movies.size()); 

        //printf("before\t"); 
        //sbefore = approx_log_likelihood_stochastic(); 
        
        if (_env.use_docs && !_env.fixed_doc_param) {

            // local step (update phi's)
            for (MovieMap::const_iterator itr = _sampled_movies.begin();
                 itr != _sampled_movies.end(); ++itr) { 

                uint32_t nd = itr->first;

                const WordVec *w = _ratings.get_words(nd);
                for (uint32_t nw = 0; w && nw < w->size(); nw++) {
                    WordCount p = (*w)[nw];
                    uint32_t word = p.first;
                    uint32_t count = p.second;
                    sampled_words[word] = true; 

                    get_phi(_theta, nd, _beta, word, phi);

                    if (count > 1)
                        phi.scale(count);

                    _theta.update_shape_next(nd, phi);
                    _beta.update_shape_next(word, phi);
                }
            }
        }

        if (_env.use_ratings) {
            for (MovieMap::const_iterator itr = _sampled_movies.begin();
                itr != _sampled_movies.end(); ++itr) { 

                uint32_t nd = itr->first;

                const vector<uint32_t> *users = _ratings.get_users(nd);
                for (uint32_t i = 0; i < users->size(); ++i) {
                    uint32_t nu = (*users)[i];
                    yval_t y = _ratings.r(nu,nd);
                    assert (y > 0);

                    if (_env.use_docs) {
                        get_xi(nu, nd, xi, xi_a, xi_b);

                        if (y > 1) {
                            xi_a.scale(y);
                            xi_b.scale(y);
                            xi.scale(y);
                        }

                        if (!_env.fixed_doc_param)
                            _theta.update_shape_next(nd, xi_a);
                        _epsilon.update_shape_next(nd, xi_b);
                        _x.update_shape_next(nu, xi_a);
                        _x.update_shape_next(nu, xi_b);

                        if (!_env.fixeda)
                            _a.update_shape_next(nd, y); // since \sum_k \xi_k = 1

                    } else { // ratings only

                        get_phi(_x, nu, _epsilon, nd, phi);
                        if (y > 1)
                            phi.scale(y);

                        _x.update_shape_next(nu, phi);
                        _epsilon.update_shape_next(nd, phi);
                    }
                }
            }
        }

        if (_env.vb || (_env.vbinit && _iter < _env.vbinit_iter))
            stochastic_update_all_rates_in_seq(sampled_words);
        else {
            printf("not implemented\n"); 
            assert(false); 
        }

        double scale_docs = _env.ndocs / _sampled_movies.size(); // TODO: is this correct?
        //scale_docs = 0.0; 
        //printf("scale_docs %f\n", scale_docs); 

        if (_env.use_ratings) { 
            bool rate_has_been_updated = false; 
            for (UserMap::const_iterator itr = _sampled_users.begin();
                    itr != _sampled_users.end(); ++itr) { 
                uint32_t nu = itr->first;
                if (_env.vb || (_env.vbinit && _iter < _env.vbinit_iter)) {

                    Array scurr(_k), rcurr(_k);
                    _rhou[nu] = pow(_tau0 + _userc[nu], - _kappa);

                    for (uint32_t k = 0; k < _k; ++k) {
                        scurr[k] = (1 - _rhou[nu]) * _x.shape_curr().const_data()[nu][k] +	
                                       _rhou[nu] * (_x.sprior() + scale_docs * _x.shape_next().const_data()[nu][k]);
                        if(!rate_has_been_updated)  { 
                            rcurr[k] = (1 - _rhou[nu]) * _x.rate_curr().const_data()[k] +	
                                           _rhou[nu] * (_x.rprior() + scale_docs * _x.rate_next().const_data()[k]);
                        }
                   }
                   _x.set_shape_curr(nu, scurr); 
                   if(!rate_has_been_updated) { // rate is shared across all users
                       _x.set_rate_curr(rcurr); 
                       rate_has_been_updated = true; 
                   }
                } else {
                    // TODO: implement non-vb
                    printf("not implemented\n"); 
                    assert(false);
                }
                ++(_userc[nu]);
            }
            _x.compute_expectations(); 
        }

        if (_env.use_docs && !_env.fixed_doc_param) {

            bool rate_has_been_updated = false; 
            for (UserMap::const_iterator itr = sampled_words.begin();
                    itr != sampled_words.end(); ++itr) {
                uint32_t nw = itr->first;
                if (_env.vb || (_env.vbinit && _iter < _env.vbinit_iter)) {
                    Array scurr(_k), rcurr(_k);
                    _rhow[nw] = pow(_tau0 + _wordc[nw], - _kappa);

                    for (uint32_t k = 0; k < _k; ++k) {
                        scurr[k] = (1 - _rhow[nw]) * _beta.shape_curr().const_data()[nw][k] +	
                                       _rhow[nw] * (_beta.sprior() + scale_docs * _beta.shape_next().const_data()[nw][k]);
                        if(!rate_has_been_updated)  { 
                            rcurr[k] = (1 - _rhow[nw]) * _beta.rate_curr().const_data()[k] +	
                                           _rhow[nw] * (_beta.rprior() + scale_docs * _beta.rate_next().const_data()[k]);
                        }
                   }
                   _beta.set_shape_curr(nw, scurr); 
                   if(!rate_has_been_updated) { // rate is shared across all words
                       _beta.set_rate_curr(rcurr); 
                       rate_has_been_updated = true; 
                   }
                } else
                    assert(false); 
                ++(_wordc[nw]);
            }
            _beta.compute_expectations(); 
        }



        if (_iter % 10 == 0 || _iter+1 >= _env.max_iterations) {
            lerr("Iteration %d\n", _iter);
            approx_log_likelihood();
            if (_env.use_ratings) {
                if (!sample_all) { 
                    compute_likelihood(false);
                    if (!compute_likelihood(true)) { 
                        sample_all = true; 
                        printf("sampling all documents\n"); 
                    }
                } else {
                    save_model();
                    exit(0); 
                }
            }
            if (_iter % 50 == 0 || _iter+1 >= _env.max_iterations) 
                save_model();
        }
        if (_env.save_state_now) { 
            if (sample_all)
                exit(0);
            else
                sample_all = true; 
        } 

        if (!sample_all)
            ++_iter;
        approx_log_likelihood();

        if(_iter >= _env.max_iterations) { 
            if (sample_all) { 
                printf("DONE (%d iterations)\n", _iter); 
                exit(0); 
            } else
                sample_all = true; 
        }

        //printf("after\t"); safter = approx_log_likelihood_stochastic(); 
        //if(safter < sbefore) assert(false); 
    }
}
void
CollabTM::update_all_rates()
{
  if (_env.use_docs && !_env.fixed_doc_param) {
    // update theta rate
    Array betasum(_k);
    _beta.sum_rows(betasum);
    _theta.update_rate_next(betasum);
    
    // update beta rate
    Array thetasum(_k);
    _theta.sum_rows(thetasum);
    _beta.update_rate_next(thetasum);
  }

  if (_env.use_ratings) { 
    // update theta rate
    Array xsum(_k);
    _x.sum_rows(xsum);
    
    if (!_env.fixed_doc_param)
      if (!_env.fixeda)
	_theta.update_rate_next(xsum, _a.expected_v());
      else
	_theta.update_rate_next(xsum);
    
    // update x rate
    if (_env.use_docs) {
      Array scaledthetasum(_k);
      if (!_env.fixeda)
	_theta.scaled_sum_rows(scaledthetasum, _a.expected_v());
      else
	_theta.sum_rows(scaledthetasum);
      _x.update_rate_next(scaledthetasum);
    }

    Array scaledepsilonsum(_k);
    if (!_env.fixeda)
      _epsilon.scaled_sum_rows(scaledepsilonsum, _a.expected_v());
    else
      _epsilon.sum_rows(scaledepsilonsum);
    _x.update_rate_next(scaledepsilonsum);
    
    // update epsilon rate
    if (_env.fixeda)
      _epsilon.update_rate_next(xsum);
    else
      _epsilon.update_rate_next(xsum, _a.expected_v());

    if (!_env.fixeda ) {
      // update 'a' rate
      Array arate(_ndocs);
      Matrix &theta_ev = _theta.expected_v();
      const double **theta_evd = theta_ev.const_data();
      Matrix &epsilon_ev = _epsilon.expected_v();
      const double **epsilon_evd = epsilon_ev.const_data();
      for (uint32_t nd = 0; nd < _ndocs; ++nd)
	for (uint32_t k = 0; k < _k; ++k)
	  arate[nd] += xsum[k] * (theta_evd[nd][k] + epsilon_evd[nd][k]);
      _a.update_rate_next(arate);
    }
  }
}

void
CollabTM::stochastic_update_all_rates_in_seq(UserMap &sampled_words)
{

  Array ta(_ndocs);
  ta.set_elements(1); 

  Array &a_ev = _a.expected_v();
  const double *a_evd = a_ev.const_data();

  if (!_env.fixeda) 
      for (int i=0; i < _ndocs; ++i)
          ta[i] *= a_evd[i]; 

  if (_env.use_docs && !_env.fixed_doc_param) {
    // update theta rate
    Array betasum(_k);
    _beta.sum_rows(betasum, sampled_words);
    _theta.update_rate_next(betasum);
  }
  

  Array xsum(_k);
  if (_env.use_ratings) {
    _x.sum_rows(xsum, _sampled_users);
    if (!_env.fixed_doc_param) {
          if (_env.fixeda)
            _theta.update_rate_next(xsum, ta);
          else
            _theta.update_rate_next(xsum, _a.expected_v());
    }
  }
  
  if (!_env.fixed_doc_param) {
     _theta.update_curr(_sampled_movies);
     _theta.compute_expectations();
  }

  if (_env.use_docs && !_env.fixed_doc_param) {
    // update beta rate
    Array thetasum(_k);
    _theta.sum_rows(thetasum, _sampled_movies);
    _beta.update_rate_next(thetasum);

    //_beta.update_curr(sampled_words); 
    //_beta.compute_expectations();
  }
 
    if (_env.use_ratings) {
    // update x rate
    Array scaledthetasum(_k);
    if (!_env.fixeda)
        _theta.scaled_sum_rows(scaledthetasum, _a.expected_v(), _sampled_movies);
    else
        _theta.sum_rows(scaledthetasum, _sampled_movies);
    
    Array scaledepsilonsum(_k);
    if (!_env.fixeda)
      _epsilon.scaled_sum_rows(scaledepsilonsum, _a.expected_v(), _sampled_movies);
    else
      _epsilon.sum_rows(scaledepsilonsum, _sampled_movies);
    
    _x.update_rate_next(scaledthetasum);
    _x.update_rate_next(scaledepsilonsum);

    //_x.swap();
    //_x.compute_expectations();
    
    // update epsilon rate
    if (_env.fixeda)
      _epsilon.update_rate_next(xsum);
    else
      _epsilon.update_rate_next(xsum, _a.expected_v());
    
    _epsilon.update_curr(_sampled_movies);
    _epsilon.compute_expectations();
 
    if (!_env.fixeda) {
      // update 'a' rate
      Array arate(_ndocs);
      Matrix &theta_ev = _theta.expected_v();
      const double **theta_evd = theta_ev.const_data();
      Matrix &epsilon_ev = _epsilon.expected_v();
      const double **epsilon_evd = epsilon_ev.const_data();
      uint32_t nd; 
      for (UserMap::const_iterator itr = _sampled_movies.begin();
              itr != _sampled_movies.end(); ++itr) { 
          nd = itr->first; 
          for (uint32_t k = 0; k < _k; ++k)
            arate[nd] += xsum[k] * (theta_evd[nd][k] + epsilon_evd[nd][k]);
      }
      _a.update_rate_next(arate);

      _a.update_curr(_sampled_movies);
      _a.compute_expectations();
    }
  }
}


void
CollabTM::update_all_rates_in_seq()
{
  if (_env.use_docs && !_env.fixed_doc_param) {
    // update theta rate
    Array betasum(_k);
    _beta.sum_rows(betasum);
    _theta.update_rate_next(betasum);
  }
  
  Array xsum(_k);
  if (_env.use_ratings) {
    _x.sum_rows(xsum);
    if (!_env.fixed_doc_param)
      if (_env.fixeda)
	_theta.update_rate_next(xsum);
      else
	_theta.update_rate_next(xsum, _a.expected_v());
  }
  
  if (!_env.fixed_doc_param) {
    _theta.swap();
    _theta.compute_expectations();
  }

  if (_env.use_docs && !_env.fixed_doc_param) {
    // update beta rate
    Array thetasum(_k);
    _theta.sum_rows(thetasum);
    _beta.update_rate_next(thetasum);

    _beta.swap();
    _beta.compute_expectations();
  }
  
  if (_env.use_ratings) {
    // update x rate
    Array scaledthetasum(_k);
    if (!_env.fixeda)
      _theta.scaled_sum_rows(scaledthetasum, _a.expected_v());
    else
      _theta.sum_rows(scaledthetasum);
    
    Array scaledepsilonsum(_k);
    if (!_env.fixeda)
      _epsilon.scaled_sum_rows(scaledepsilonsum, _a.expected_v());
    else
      _epsilon.sum_rows(scaledepsilonsum);
    
    _x.update_rate_next(scaledthetasum);
    _x.update_rate_next(scaledepsilonsum);
    
    _x.swap();
    _x.compute_expectations();
    
    // update epsilon rate
    if (_env.fixeda)
      _epsilon.update_rate_next(xsum);
    else
      _epsilon.update_rate_next(xsum, _a.expected_v());
    
    _epsilon.swap();
    _epsilon.compute_expectations();
    
    if (!_env.fixeda) {
      // update 'a' rate
      Array arate(_ndocs);
      Matrix &theta_ev = _theta.expected_v();
      const double **theta_evd = theta_ev.const_data();
      Matrix &epsilon_ev = _epsilon.expected_v();
      const double **epsilon_evd = epsilon_ev.const_data();
      for (uint32_t nd = 0; nd < _ndocs; ++nd)
	for (uint32_t k = 0; k < _k; ++k)
	  arate[nd] += xsum[k] * (theta_evd[nd][k] + epsilon_evd[nd][k]);
      _a.update_rate_next(arate);
      _a.swap();
      _a.compute_expectations();
    }
  }
}

void
CollabTM::swap_all()
{
  if (_env.use_docs && !_env.fixed_doc_param) {
    _theta.swap();
    _beta.swap();
  }
  if (_env.use_ratings) {
    _epsilon.swap();
    if (!_env.fixeda)
      _a.swap();
    _x.swap();
  }
}

void
CollabTM::compute_all_expectations()
{
  if (_env.use_docs && !_env.fixed_doc_param) { 
    _theta.compute_expectations();
    _beta.compute_expectations();
  }
  
  if (_env.use_ratings) {
    _epsilon.compute_expectations();
    if (!_env.fixeda)
      _a.compute_expectations();
    _x.compute_expectations();
  }
}

void
CollabTM::approx_log_likelihood()
{
  return; // XXX
  if (_nusers > 10000 || _k > 10)
    return;

  const double ** etheta = _theta.expected_v().const_data();
  const double ** elogtheta = _theta.expected_logv().const_data();
  const double ** ebeta = _beta.expected_v().const_data();
  const double ** elogbeta = _beta.expected_logv().const_data();
  const double ** ex = _x.expected_v().const_data();
  const double ** elogx = _x.expected_logv().const_data();
  const double ** eepsilon = _epsilon.expected_v().const_data();
  const double ** elogepsilon = _epsilon.expected_logv().const_data();
  
  const double *ea = _env.fixeda? NULL : _a.expected_v().const_data();
  const double *eloga = _env.fixeda ? NULL : _a.expected_logv().const_data();

  double s = .0;
  Array phi(_k);
  Array xi(2*_k);
  Array xi_a(_k);
  Array xi_b(_k);

  for (uint32_t nd = 0; nd < _ndocs; ++nd) {
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

  debug("E1: s = %f\n", s);

  for (uint32_t nu = 0; nu < _nusers; ++nu) {
    const vector<uint32_t> *docs = _ratings.get_movies(nu);
    
    for (uint32_t j = 0; j < docs->size(); ++j) {
      uint32_t nd = (*docs)[j];
      yval_t y = _ratings.r(nu,nd);
      
      assert (y > 0);
      
      get_xi(nu, nd, xi, xi_a, xi_b);

      debug("xi = %s\n", xi.s().c_str());

      double v = .0;
      for (uint32_t k = 0; k < 2*_k; ++k) {
	double r = .0;
	if (k < _k)
	  r = !_env.fixeda ? (elogx[nu][k] + elogtheta[nd][k] + eloga[nd]) : 
	    (elogx[nu][k] + elogtheta[nd][k]);
	else {
	  uint32_t t = k - _k;
	  r = !_env.fixeda ? (elogx[nu][t] + elogepsilon[nd][t] + eloga[nd]) :
	    (elogx[nu][t] + elogepsilon[nd][t]);
	}
	v += y * xi[k] * (r - log(xi[k]));
      }
      s += v;

      for (uint32_t k = 0; k < 2*_k; ++k) {
	double r = .0;
	if (k < _k)
	  r = !_env.fixeda ? (ex[nu][k] * etheta[nd][k] * ea[nd]) : \
	    (ex[nu][k] * etheta[nd][k]);
	else {
	  uint32_t t = k - _k;
	  r = !_env.fixeda ? (ex[nu][t] * eepsilon[nd][t] * ea[nd]) : \
	    (ex[nu][t] * eepsilon[nd][t]);
	}
	s -= r;
      }
    }
  }

  debug("E2: s = %f\n", s);

  s += _theta.compute_elbo_term();
  s += _beta.compute_elbo_term();
  s += _x.compute_elbo_term();
  s += _epsilon.compute_elbo_term();
  if (!_env.fixeda)
    s += _a.compute_elbo_term();

  debug("E3: s = %f\n", s);

  fprintf(_af, "%.5f\n", s);
  fflush(_af);
}

void
CollabTM::save_model()
{
  IDMap idmap; // null map
  if (_env.use_ratings) {
    printf("saving ratings state\n");
    fflush(stdout);
    _x.save_state(_ratings.seq2user());
    _epsilon.save_state(_ratings.seq2movie());
    if (!_env.fixeda) 
      _a.save_state(_ratings.seq2movie());
  }

  if (_env.use_docs) {
    _theta.save_state(_ratings.seq2movie());
    _beta.save_state(idmap);
  }
}

void
CollabTM::ppc()
{
  printf("loading theta\n");
  _theta.load();
  printf("done\n");
  fflush(stdout);
  printf("loading beta\n");
  _beta.load();
  printf("done\n");
  fflush(stdout);
  //_epsilon.load();
  //_x.load();

  const double **theta_shape_curr = _theta.shape_curr().const_data();
  const double **theta_rate_curr = _theta.rate_curr().const_data();
  const double **beta_shape_curr = _beta.shape_curr().const_data();
  const double *beta_rate_curr = _beta.rate_curr().const_data();

  // generate data for users and documents
  char buf[1024];
  sprintf(buf, "ppc.tsv");
  FILE *f = fopen(buf, "w");
    
  for (uint32_t nd = 0; nd < _ndocs; ++nd) {
    const WordVec *w = _ratings.get_words(nd);

    uint32_t maxword = 0;
    uint32_t maxcount = 0;
    for (uint32_t nw = 0; w && nw < w->size(); nw++) {
      WordCount p = (*w)[nw];
      uint32_t word = p.first;
      uint32_t count = p.second;  
      if (count > maxcount) {
	maxcount = count;
	maxword = word;
      }
    }

    uint32_t nw = maxword;
    fprintf(f, "%d\t%d\t%d\t%d\n", nd, nw, 0, maxcount);
    for (uint32_t rep = 1; rep < 1000; rep++) {
      uint32_t yrep = .0;
      double s = .0;
      for (uint32_t k = 0; k < _k; k++) {
	if (beta_rate_curr[k] > .0 && theta_rate_curr[nd][k] > .0) {
	  double rtheta = gsl_ran_gamma(_r, theta_shape_curr[nd][k], 
					1 / theta_rate_curr[nd][k]);
	  double rbeta = gsl_ran_gamma(_r, beta_shape_curr[nw][k],
				       1 / beta_rate_curr[k]);
	  s += rtheta * rbeta;
	}
      }
      
      lerr("s = %f", s);
      yrep = gsl_ran_poisson(_r, s);
      lerr("rep = %d, s = %f, yrep = %d", rep, s, yrep);
      lerr("rep = %d, yrep = %d, yobs = %d, word = %d\n", rep, yrep, maxcount, maxword);
      fprintf(f, "%d\t%d\t%d\t%d\n", nd, nw, rep, yrep);
    }
    if (nd > 100)
      break;
  }
  fclose(f);
}

bool
CollabTM::compute_likelihood(bool validation)
{
  assert (_env.use_ratings);

  uint32_t k = 0, kzeros = 0, kones = 0;
  double s = .0, szeros = 0, sones = 0;
  
  CountMap *mp = NULL;
  FILE *ff = NULL;
  if (validation) {
    mp = &_validation_map;
    ff = _vf;
  } else {
    mp = &_test_map;
    ff = _tf;
  }

  for (CountMap::const_iterator i = mp->begin();
       i != mp->end(); ++i) {
    const Rating &e = i->first;
    uint32_t n = e.first;
    uint32_t m = e.second;

    yval_t r = i->second;
    double u = per_rating_likelihood(n,m,r);
    s += u;
    k += 1;
  }

  double a = .0;
  info("s = %.5f\n", s);
  fprintf(ff, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(ff);
  a = s / k;  
  
  if (!validation)
    return true;
  
  bool stop = false;
  int why = -1;
  if (_iter > 10) {
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
    save_model();
    exit(0);
    //return false; 
  }
  return true;
}

void
CollabTM::precision()
{ 
  double mhits10 = 0, mhits100 = 0;
  double cumndcg10 = 0, cumndcg100 = 0;
  uint32_t total_users = 0;
  FILE *f = 0;
  if (_save_ranking_file)
    f = fopen(Env::file_str("/ranking.tsv").c_str(), "w");
  
  if (!_save_ranking_file) {
    _sampled_users.clear();
    do {
      uint32_t n = gsl_rng_uniform_int(_r, _nusers);
      _sampled_users[n] = true;
    } while (_sampled_users.size() < 1000 && _sampled_users.size() < _nusers / 2);
  }
  
  KVArray mlist(_ndocs);
  KVIArray ndcglist(_ndocs);
  uint32_t cc = 0;
  for (UserMap::const_iterator itr = _sampled_users.begin();
       itr != _sampled_users.end(); ++itr, ++cc) {
      uint32_t n = itr->first;

      for (uint32_t m = 0; m < _ndocs; ++m) {
	Rating r(n,m);
	if (_ratings.r(n,m) > 0 || is_validation(r)) { // skip training
	  mlist[m].first = m;
	  mlist[m].second = .0;
	  ndcglist[m].first = m;
	  ndcglist[m].second = 0;
	  continue;
	}
	
	double pred = per_rating_prediction(n, m); 
	mlist[m].first = m;
	mlist[m].second = pred;
	ndcglist[m].first = m;
	CountMap::const_iterator itr = _test_map.find(r);
	if (itr != _test_map.end()) {
	  ndcglist[m].second = itr->second;
	} else { 
	  ndcglist[m].second = 0;
	}
      }

      uint32_t hits10 = 0, hits100 = 0;
      double   dcg10 = .0, dcg100 = .0; 
      uint32_t n2 = 0; 
      mlist.sort_by_value();
      for (uint32_t j = 0; j < mlist.size() && j < _topN_by_user; ++j) {
          KV &kv = mlist[j];
          uint32_t m = kv.first;
          double pred = kv.second;
          Rating r(n, m);

          uint32_t m2 = 0;
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
              int v_ = itr->second;
              int v = _ratings.rating_class(v_);
              assert(v > 0);
              if (_save_ranking_file) {
                  if (_ratings.r(n, m) == .0) // skip training
                      fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, v);
              }


              if (j < 10) {
                  hits10++;
                  hits100++;
		  dcg10 += (pow(2.,v_) - 1)/log(j+2);
		  dcg100 += (pow(2.,v_) - 1)/log(j+2);
              } else if (j < 100) {
                  hits100++;
                  dcg100 += (pow(2.,v_) - 1)/log(j+2);
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

      // DCG normalizer
      double dcg10_gt = 0, dcg100_gt = 0;
      bool user_has_test_ratings = true; 
      ndcglist.sort_by_value();
      for (uint32_t j = 0; j < ndcglist.size() && j < _topN_by_user; ++j) {
          int v = ndcglist[j].second; 
          if(v==0) { // all subsequent docs are irrelevant
            if(j==0)
                user_has_test_ratings = false; 
            break;
          }

          if (j < 10) { 
              dcg10_gt += (pow(2.,v) - 1)/log(j+2);
              dcg100_gt += (pow(2.,v) - 1)/log(j+2);
          } else if (j < 100) {
              dcg100_gt += (pow(2.,v) - 1)/log(j+2);
          }
      }
      if(user_has_test_ratings) { 
          cumndcg10 += dcg10/dcg10_gt;
          cumndcg100 += dcg100/dcg100_gt;
      } 
      //fprintf(_df, "%d (n2=%d): %.5f\t%.5f\n", 
      //    n,
      //    n2,
      //    dcg10,
      //    dcg10_gt);
      printf("\r%d", cc);
      fflush(stdout);
  }
  if (_save_ranking_file)
    fclose(f);
  fprintf(_pf, "%.5f\t%.5f\n", 
  	  mhits10 / total_users, 
  	  mhits100 / total_users);
  fflush(_pf);
  fprintf(_df, "%.5f\t%.5f\n", 
  	  cumndcg10 / total_users, 
  	  cumndcg100 / total_users);
  fflush(_df);
}


void
CollabTM::coldstart_precision()
{ 
  double mhits10 = 0, mhits100 = 0;
  double cumndcg10 = 0, cumndcg100 = 0;
  uint32_t total_users = 0;
  FILE *f = 0;
  if (_save_ranking_file)
    f = fopen(Env::file_str("/coldstart_ranking.tsv").c_str(), "w");
  
  KVArray mlist(_ndocs);
  KVIArray ndcglist(_ndocs);
  uint32_t cc = 0;  
  for (uint32_t itr = 0; itr < _cs_test_users->size() && itr < 10000; itr++, cc++) {
    uint32_t n =(*_cs_test_users)[itr];
    
    // now the items set consists of only those in the "coldstart" list
    for (MovieMap::const_iterator i = _cold_start_docs.begin(); 
	 i != _cold_start_docs.end(); ++i) {
      uint32_t m = i->first;
      Rating r(n,m);
      //
      // none of these ratings could have appeared in training
      // so we rank all of them
      //
      double pred = coldstart_per_rating_prediction(n, m); 
      mlist[m].first = m;
      mlist[m].second = pred;
      ndcglist[m].first = m;
      CountMap::const_iterator itr = _coldstart_test_map.find(r);
      if (itr != _coldstart_test_map.end()) {
	ndcglist[m].second = itr->second;
      } else { 
	ndcglist[m].second = 0;
      }
    }
    
    uint32_t hits10 = 0, hits100 = 0;
    double   dcg10 = .0, dcg100 = .0; 
    uint32_t n2 = 0; 
    mlist.sort_by_value();
    for (uint32_t j = 0; j < mlist.size() && j < _topN_by_user; ++j) {
      KV &kv = mlist[j];
      uint32_t m = kv.first;
      double pred = kv.second;
      Rating r(n, m);

      uint32_t m2 = 0;
      if (_save_ranking_file) {
	IDMap::const_iterator it = _ratings.seq2user().find(n);
	assert (it != _ratings.seq2user().end());

	IDMap::const_iterator mt = _ratings.seq2movie().find(m);
	if (mt == _ratings.seq2movie().end())
	  continue;

	m2 = mt->second;
	n2 = it->second;
      }

      CountMap::const_iterator itr = _coldstart_test_map.find(r);
      if (itr != _coldstart_test_map.end()) {
	int v_ = itr->second;
	int v = _ratings.rating_class(v_);
	assert(v > 0);
	if (_save_ranking_file) {
	  if (_ratings.r(n, m) == .0) // skip training
	    fprintf(f, "%d\t%d\t%.5f\t%d\n", n2, m2, pred, v);
	}


	if (j < 10) {
	  hits10++;
	  hits100++;
	  dcg10 += (pow(2.,v_) - 1)/log(j+2);
	  dcg100 += (pow(2.,v_) - 1)/log(j+2);
	} else if (j < 100) {
	  hits100++;
	  dcg100 += (pow(2.,v_) - 1)/log(j+2);
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

    // DCG normalizer
    double dcg10_gt = 0, dcg100_gt = 0;
    bool user_has_test_ratings = true; 
    ndcglist.sort_by_value();
    for (uint32_t j = 0; j < ndcglist.size() && j < _topN_by_user; ++j) {
      int v = ndcglist[j].second; 
      if(v==0) { // all subsequent docs are irrelevant
	if(j==0)
	  user_has_test_ratings = false; 
	break;
      }

      if (j < 10) { 
	dcg10_gt += (pow(2.,v) - 1)/log(j+2);
	dcg100_gt += (pow(2.,v) - 1)/log(j+2);
      } else if (j < 100) {
	dcg100_gt += (pow(2.,v) - 1)/log(j+2);
      }
    }
    if(user_has_test_ratings) { 
      cumndcg10 += dcg10/dcg10_gt;
      cumndcg100 += dcg100/dcg100_gt;
    } 
    //fprintf(_df, "%d (n2=%d): %.5f\t%.5f\n", 
    //    n,
    //    n2,
    //    dcg10,
    //    dcg10_gt);
    printf("\r%d", cc);
    fflush(stdout);
  }
  if (_save_ranking_file)
    fclose(f);
  fprintf(_cs_pf, "%.5f\t%.5f\n", 
  	  mhits10 / total_users, 
  	  mhits100 / total_users);
  fflush(_cs_pf);
  fprintf(_cs_df, "%.5f\t%.5f\n", 
  	  cumndcg10 / total_users, 
  	  cumndcg100 / total_users);
  fflush(_cs_df);
}


double
CollabTM::per_rating_prediction(uint32_t user, uint32_t doc) const
{
  const double ** etheta = _theta.expected_v().const_data();
  //const double ** elogtheta = _theta.expected_logv().const_data();
  const double ** ebeta = _beta.expected_v().const_data();
  //const double ** elogbeta = _beta.expected_logv().const_data();
  
  const double ** ex = _x.expected_v().const_data();
  //const double ** elogx = _x.expected_logv().const_data();
  const double ** eepsilon = _epsilon.expected_v().const_data();
  //const double ** elogepsilon = _epsilon.expected_logv().const_data();
  
  const double *ea = _env.fixeda? NULL : _a.expected_v().const_data();
  //const double *eloga = _env.fixeda ? NULL : _a.expected_logv().const_data();

  double s = .0;
  double item_contrib = 0;
  for (uint32_t k = 0; k < _k; ++k) {
    item_contrib = (etheta[doc][k] + eepsilon[doc][k]); 
    if (!_env.fixeda)
      s +=  item_contrib * ea[doc] * ex[user][k];
    else
      s += item_contrib * ex[user][k];
  }
    
  if (s < 1e-30)
    s = 1e-30;

  return s;
}

double
CollabTM::coldstart_per_rating_prediction(uint32_t user, uint32_t doc) const
{
  assert (_env.fixeda);
  const double ** etheta = _cstheta.expected_v().const_data();
  const double ** ex = _x.expected_v().const_data();
  const double ** eepsilon = _epsilon.expected_v().const_data();
  
  IDMap::const_iterator itr = _doc_to_cs_idmap.find(doc);
  assert (itr != _doc_to_cs_idmap.end());
  uint32_t docseq = itr->second;
  
  double s = .0;
  double item_contrib = 0;
  for (uint32_t k = 0; k < _k; ++k) {
    item_contrib = (etheta[docseq][k] + eepsilon[doc][k]);
    s += item_contrib * ex[user][k];
  }
  if (s < 1e-30)
    s = 1e-30;
  return s;
}


double
CollabTM::per_rating_likelihood(uint32_t user, uint32_t doc, yval_t y, 
				bool coldstart) const
{
  assert (_env.use_ratings);

  double s = coldstart? coldstart_per_rating_prediction(user, doc) : 
    per_rating_prediction(user, doc); 
  info("%d, %d, s = %f, f(y) = %ld\n", p, q, s, factorial(y));
  
  double v = .0;
  v = y * log(s) - s - log_factorial(y);
  return v;
}

uint32_t
CollabTM::factorial(uint32_t n)  const
{ 
  //return n <= 1 ? 1 : (n * factorial(n-1));
  uint32_t v = 1;
  for (uint32_t i = 2; i <= n; ++i)
    v *= i;
  return v;
} 

double
CollabTM::log_factorial(uint32_t n)  const
{ 
  //return n <= 1 ? 1 : (n * factorial(n-1));
  double v = log(1);
  for (uint32_t i = 2; i <= n; ++i)
    v += log(i);
  return v;
} 

double
CollabTM::coldstart_local_inference()
{
  // keep _beta fixed and compute theta for coldstart docs
  _cstheta.set_to_prior();
  _cstheta.set_to_prior_curr();

  uint32_t processed_docs = 0;
  for (MovieMap::const_iterator i = _cold_start_docs.begin(); 
       i != _cold_start_docs.end(); ++i) {
    uint32_t doc = i->first;
    
    assert (_doc_to_cs_idmap.find(doc) != _doc_to_cs_idmap.end());
    uint32_t docseq = _doc_to_cs_idmap[doc];
    
    Array phi(_k);
    const WordVec *w = _ratings.get_words(doc);
    for (uint32_t nw = 0; w && nw < w->size(); nw++) {
      WordCount p = (*w)[nw];
      uint32_t word = p.first;
      uint32_t count = p.second;
      
      get_phi(_cstheta, docseq, _beta, word, phi);
      
      if (count > 1)
	phi.scale(count);
      
      _cstheta.update_shape_next(docseq, phi);
      _beta.update_shape_next(word, phi);
    }
    processed_docs++;
    printf("\r%d", processed_docs);
    fflush(stdout);
  }
  Env::plog("processed %d cold-start docs", processed_docs);

  Array betasum(_k);
  _beta.sum_rows(betasum);
  _cstheta.update_rate_next(betasum);
  _cstheta.swap();
  _cstheta.compute_expectations();

  // save it
  _cstheta.save_state(_ratings.seq2movie());
}

double
CollabTM::coldstart_rating_likelihood()
{
  lerr("computing coldstart local thetas");
  coldstart_local_inference();
  
  lerr("computing coldstart rating likelihood");
  // compute likelihood of "heldout" ratings
  // ALL ratings for a coldstart document are heldout

  double s = .0;
  uint32_t k = 0;
  for (MovieMap::const_iterator i = _cold_start_docs.begin(); 
       i != _cold_start_docs.end(); ++i) {
    double ss = .0;
    uint32_t kk = 0;
    uint32_t nd = i->first;
    const vector<uint32_t> *users = _ratings.get_users(nd);

    // need this to access _cstheta
    assert (_doc_to_cs_idmap.find(nd) != _doc_to_cs_idmap.end());
    uint32_t docseq = _doc_to_cs_idmap[nd];
    debug("heldout doc %d seq %d", nd, docseq);
    
    for (uint32_t u = 0; u < users->size(); ++u) {
      uint32_t nu = (*users)[u];
      yval_t y = _ratings.r(nu,nd);
      assert (y > 0);

      // note: need docseq whenever looking up "cstheta"
      double u = per_rating_likelihood(nu, nd, y, true);
      s += u;
      k += 1;
      ss += u;
      kk += 1;
    }
    
    fprintf(_cs_tf2, "%d\t%d\t%.9f\t%d\n", nd, docseq, ss / kk, kk);
    fflush(_cs_tf2);
  }
  
  fprintf(_cs_tf, "%d\t%d\t%.9f\t%d\n", _iter, duration(), s / k, k);
  fflush(_cs_tf);
  lerr("computing coldstart likelihood done");
}
