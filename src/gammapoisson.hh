#ifndef GAMMAPOISSON_HH
#define GAMMAPOISSON_HH

//
// the variational posterior Gamma, the expectation of the Poisson under this
// distribution
//

template <class T>
class GPBase {
public:
  GPBase(string name = ""): _name(name) { }
  virtual ~GPBase() { }
  virtual const T &expected_v() const = 0;
  virtual const T &expected_logv() const = 0;
  virtual uint32_t n() const = 0;
  virtual uint32_t k() const = 0;
  virtual double compute_elbo_term_helper() const = 0;
  virtual void save_state(const IDMap &m) const = 0;
  void  make_nonzero(double av, double bv,
		     double &a, double &b) const;
  string name() const { return _name; }
  double compute_elbo_term() const;
private:
  string _name;
};

template<class T> inline  void
GPBase<T>::make_nonzero(double av, double bv,
			double &a, double &b) const
{
  assert (av >= 0 && bv >= 0);
  if (!(bv > .0)) 
    b = 1e-5;
  else
    b = bv;
  if (!(av > .0)) 
    a = 1e-5;
  else
    a = av;
}

template<class T> inline  double
GPBase<T>::compute_elbo_term() const
{
  double s = compute_elbo_term_helper();
  debug("sum of %s elbo terms = %f\n", name().c_str(), s);
  return s;
}

class GPMatrix : public GPBase<Matrix> {
public:
  GPMatrix(string name, double a, double b,
	   uint32_t n, uint32_t k,
	   gsl_rng **r): 
    GPBase(name),
    _n(n), _k(k),
    _sprior(a), // shape 
    _rprior(b), // rate
    _scurr(n,k),
    _snext(n,k),
    _rnext(n,k),
    _rcurr(n,k),
    _Ev(n,k),
    _Elogv(n,k),
    _r(r) { }
  virtual ~GPMatrix() {} 

  uint32_t n() const { return _n;}
  uint32_t k() const { return _k;}

  void save() const;
  
  const Matrix &shape_curr() const         { return _scurr; }
  const Matrix &rate_curr() const          { return _rcurr; }
  const Matrix &shape_next() const         { return _snext; }
  const Matrix &rate_next() const          { return _rnext; }
  const Matrix &expected_v() const         { return _Ev;    }
  const Matrix &expected_logv() const      { return _Elogv; }
  
  Matrix &shape_curr()       { return _scurr; }
  Matrix &rate_curr()        { return _rcurr; }
  Matrix &shape_next()       { return _snext; }
  Matrix &rate_next()        { return _rnext; }
  Matrix &expected_v()       { return _Ev;    }
  Matrix &expected_logv()    { return _Elogv; }

  const double sprior() const { return _sprior; }
  const double rprior() const { return _rprior; }

  void set_to_prior();
  void update_shape_next(uint32_t n, const Array &sphi);
  void update_rate_next(const Array &u, const Array &scale);
  void update_rate_next(const Array &u);

  void swap();
  void compute_expectations();
  void sum_rows(Array &v);
  void scaled_sum_rows(Array &v, const Array &scale);
  void initialize();
  void save_state(const IDMap &m) const;

  double compute_elbo_term_helper() const;

private:
  uint32_t _n;
  uint32_t _k;	
  gsl_rng **_r;
  double _sprior;
  double _rprior;

  Matrix _scurr;      // current variational shape posterior 
  Matrix _snext;      // to help compute gradient update
  Matrix _rcurr;      // current variational rate posterior (global)
  Matrix _rnext;      // help compute gradient update
  Matrix _Ev;         // expected weights under variational
		      // distribution
  Matrix _Elogv;      // expected log weights 
};

inline void
GPMatrix::set_to_prior()
{
  _snext.set_elements(_sprior);
  _rnext.set_elements(_rprior);
}

inline void
GPMatrix::update_shape_next(uint32_t n, const Array &sphi)
{
  _snext.add_slice(n, sphi);
  //printf("snext = %s\n", _snext.s().c_str());
}

inline void
GPMatrix::update_rate_next(const Array &u, const Array &scale)
{
  Array t(_k);
  for (uint32_t i = 0; i < _n; ++i) {
    for (uint32_t k = 0; k < _k; ++k)
      t[k] = u[k] *  scale[i];
    _rnext.add_slice(i, t);
  }
}

inline void
GPMatrix::update_rate_next(const Array &u)
{
  for (uint32_t i = 0; i < _n; ++i)
    _rnext.add_slice(i, u);
}

inline void
GPMatrix::swap()
{
  _scurr.swap(_snext);
  _rcurr.swap(_rnext);
  set_to_prior();
}

inline void
GPMatrix::compute_expectations()
{
  const double ** const ad = _scurr.const_data();
  const double ** const bd = _rcurr.const_data();
  double **vd1 = _Ev.data();
  double **vd2 = _Elogv.data();
  double a = .0, b = .0;
  for (uint32_t i = 0; i < _scurr.m(); ++i)
    for (uint32_t j = 0; j < _rcurr.n(); ++j) {
      make_nonzero(ad[i][j], bd[i][j], a, b);
      vd1[i][j] = a / b;
      vd2[i][j] = gsl_sf_psi(a) - log(b);
    }
}

inline void
GPMatrix::sum_rows(Array &v)
{
  const double **ev = _Ev.const_data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      v[k] += ev[i][k];
}

inline void
GPMatrix::scaled_sum_rows(Array &v, const Array &scale)
{
  assert(scale.size() == n() && v.size() == k());
  const double **ev = _Ev.const_data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      v[k] += ev[i][k] * scale[i];
}

inline void
GPMatrix::initialize()
{
  double **ad = _scurr.data();
  double **bd = _rcurr.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      ad[i][k] = _sprior + 0.001 * gsl_rng_uniform(*_r);
      bd[i][k] = _rprior + 0.1 * gsl_rng_uniform(*_r);
    }
  double **vd1 = _Ev.data();
  double **vd2 = _Elogv.data();
  
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k) {
      assert(bd[i][k]);
      vd1[i][k] = ad[i][k] / bd[i][k];
      vd2[i][k] = gsl_sf_psi(ad[i][k]) - log(bd[i][k]);
    }
} 

inline double
GPMatrix::compute_elbo_term_helper() const
{
  const double **etheta = _Ev.data();
  const double **elogtheta = _Elogv.data();
  const double ** const ad = shape_curr().const_data();
  const double ** const bd = rate_curr().const_data();

  double s = .0;
  for (uint32_t n = 0; n < _n; ++n)  {
    for (uint32_t k = 0; k < _k; ++k) {
      s += _sprior * log(_rprior) + (_sprior - 1) * elogtheta[n][k];
      s -= _rprior * etheta[n][k] + gsl_sf_lngamma(_sprior);
    }
    double a = .0, b = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      make_nonzero(ad[n][k], bd[n][k], a, b);
      s -= a * log(b) + (a - 1) * elogtheta[n][k];
      s += b * etheta[n][k] + gsl_sf_lngamma(a);
    }
  }
  return s;
}

inline void
GPMatrix::save_state(const IDMap &m) const
{
  string fname = string("/") + name() + ".tsv";
  FILE * tf = fopen(Env::file_str(fname.c_str()).c_str(), "w");
  const double **cd = expected_v().data();
  uint32_t id = 0;
  for (uint32_t i = 0; i < _n; ++i) {
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) 
      id = idt->second;
    else
      id = i;

    fprintf(tf,"%d\t", i);
    fprintf(tf,"%d\t", id);
    for (uint32_t k = 0; k < _k; ++k) {
      if (k == _k - 1)
	fprintf(tf,"%.8f\n", cd[i][k]);
      else
	fprintf(tf,"%.8f\t", cd[i][k]);
    }
  }
  fclose(tf);
}

class GPMatrixGR : public GPBase<Matrix> { // global rates
public:
  GPMatrixGR(string name, 
	     double a, double b,
	     uint32_t n, uint32_t k,
	     gsl_rng **r):
    GPBase(name),
    _n(n), _k(k),
    _sprior(a), // shape 
    _rprior(b), // rate
    _scurr(n,k),
    _snext(n,k),
    _rnext(k),
    _rcurr(k),
    _Ev(n,k),
    _Elogv(n,k),
    _r(r) { }
  virtual ~GPMatrixGR() {} 

  uint32_t n() const { return _n;}
  uint32_t k() const { return _k;}

  const Matrix &shape_curr() const         { return _scurr; }
  const Array  &rate_curr() const          { return _rcurr; }
  const Matrix &shape_next() const         { return _snext; }
  const Array  &rate_next() const          { return _rnext; }
  const Matrix &expected_v() const         { return _Ev;    }
  const Matrix &expected_logv() const      { return _Elogv; }
  
  Matrix &shape_curr()       { return _scurr; }
  Array  &rate_curr()        { return _rcurr; }
  Matrix &shape_next()       { return _snext; }
  Array  &rate_next()        { return _rnext; }
  Matrix &expected_v()       { return _Ev;    }
  Matrix &expected_logv()    { return _Elogv; }

  const double sprior() const { return _sprior; }
  const double rprior() const { return _rprior; }

  void set_to_prior();
  void update_shape_next(const Array &phi);
  void update_shape_next(uint32_t n, const Array &sphi);

  void update_rate_next(const Array &u);
  void swap();
  void compute_expectations();
  void sum_rows(Array &v);
  void scaled_sum_rows(Array &v, const Array &scale);
  void initialize();
  double compute_elbo_term_helper() const;
  void save_state(const IDMap &m) const;

private:
  uint32_t _n;
  uint32_t _k;	
  gsl_rng **_r;

  double _sprior;
  double _rprior;
  Matrix _scurr;      
  Matrix _snext;      
  Array _rcurr;       
  Array _rnext;       
  Matrix _Ev;         
  Matrix _Elogv;      
};

inline void
GPMatrixGR::set_to_prior()
{
  _snext.set_elements(_sprior);
  _rnext.set_elements(_rprior);
}

inline void
GPMatrixGR::update_shape_next(uint32_t n, const Array &sphi)
{
  _snext.add_slice(n, sphi);
}

inline void
GPMatrixGR::update_rate_next(const Array &u)
{
  _rnext += u;
}

inline void
GPMatrixGR::swap()
{
  _scurr.swap(_snext);
  _rcurr.swap(_rnext);
  set_to_prior();
}

inline void
GPMatrixGR::compute_expectations()
{
  const double ** const ad = _scurr.const_data();
  const double * const bd = _rcurr.const_data();
  double **vd1 = _Ev.data();
  double **vd2 = _Elogv.data();
  double a = .0, b = .0;
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _k; ++j) {
      make_nonzero(ad[i][j], bd[j], a, b);
      vd1[i][j] = a / b;
      vd2[i][j] = gsl_sf_psi(a) - log(b);
    }
  debug("name = %s, scurr = %s, rcurr = %s, Ev = %s\n",
	name().c_str(),
	_scurr.s().c_str(),
	_rcurr.s().c_str(),
	_Ev.s().c_str());
}

inline void
GPMatrixGR::sum_rows(Array &v)
{
  double **ev = _Ev.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      v[k] += ev[i][k];
}


inline void
GPMatrixGR::scaled_sum_rows(Array &v, const Array &scale)
{
  assert(scale.size() == n() && v.size() == k());
  const double **ev = _Ev.const_data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)
      v[k] += ev[i][k] * scale[i];
}


inline void
GPMatrixGR::initialize()
{
  double **ad = _scurr.data();
  double *bd = _rcurr.data();
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t k = 0; k < _k; ++k)  {
      ad[i][k] = _sprior + 0.001 * gsl_rng_uniform(*_r);
      bd[k] = _rprior + 0.1 * gsl_rng_uniform(*_r);
    }
  double **vd1 = _Ev.data();
  double **vd2 = _Elogv.data();
  
  for (uint32_t i = 0; i < _n; ++i)
    for (uint32_t j = 0; j < _k; ++j) {
      assert(bd[j]);
      vd1[i][j] = ad[i][j] / bd[j];
      vd2[i][j] = gsl_sf_psi(ad[i][j]) - log(bd[j]);
    }
} 

inline double
GPMatrixGR::compute_elbo_term_helper() const
{
  const double **etheta = _Ev.const_data();
  const double **elogtheta = _Elogv.const_data();
  const double ** const ad = shape_curr().const_data();
  const double * const bd = rate_curr().const_data();

  double s = .0;
  for (uint32_t n = 0; n < _n; ++n)  {
    for (uint32_t k = 0; k < _k; ++k) {
      s += _sprior * log(_rprior) + (_sprior - 1) * elogtheta[n][k];
      s -= _rprior * etheta[n][k] + gsl_sf_lngamma(_sprior);
      debug("ehelper: %f:%f:%f log:%f\n", s, etheta[n][k], gsl_sf_lngamma(_sprior), elogtheta[n][k]);
    }
    double a = .0, b = .0;
    for (uint32_t k = 0; k < _k; ++k) {
      make_nonzero(ad[n][k], bd[k], a, b);
      s -= a * log(b) + (a - 1) * elogtheta[n][k];
      s += b * etheta[n][k] + gsl_sf_lngamma(a);
    }
  }
  return s;
}

inline void
GPMatrixGR::save_state(const IDMap &m) const
{
  string fname = string("/") + name() + ".tsv";
  FILE * tf = fopen(Env::file_str(fname.c_str()).c_str(), "w");
  const double **cd = expected_v().data();
  uint32_t id = 0;
  for (uint32_t i = 0; i < _n; ++i) {
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) 
      id = idt->second;
    else
      id = i;

    fprintf(tf,"%d\t", i);
    fprintf(tf,"%d\t", id);
    for (uint32_t k = 0; k < _k; ++k) {
      if (k == _k - 1)
	fprintf(tf,"%.8f\n", cd[i][k]);
      else
	fprintf(tf,"%.8f\t", cd[i][k]);
    }
  }
  fclose(tf);
}


class GPArray : public GPBase<Array> {
public:
  GPArray(string name, 
	  double a, double b,
	  uint32_t n, gsl_rng **r): 
    GPBase(name),
    _n(n),
    _sprior(a), // shape 
    _rprior(b), // rate
    _scurr(n), _snext(n),
    _rnext(n), _rcurr(n),
    _Ev(n), _Elogv(n),
    _r(r) { }
  ~GPArray() {}

  uint32_t n() const { return _n;}
  uint32_t k() const { return 0;}

  const Array &shape_curr() const         { return _scurr; }
  const Array &rate_curr() const          { return _rcurr; }
  const Array &shape_next() const         { return _snext; }
  const Array &rate_next() const          { return _rnext; }
  const Array &expected_v() const         { return _Ev;    }
  const Array &expected_logv() const      { return _Elogv; }
  
  Array &shape_curr()       { return _scurr; }
  Array &rate_curr()        { return _rcurr; }
  Array &shape_next()       { return _snext; }
  Array &rate_next()        { return _rnext; }
  Array &expected_v()       { return _Ev;    }
  Array &expected_logv()    { return _Elogv; }

  const double sprior() const { return _sprior; }
  const double rprior() const { return _rprior; }
  
  uint32_t n() { return _n; }

  void set_to_prior();
  void update_shape_next(const Array &phi);
  void update_shape_next(uint32_t n, uint32_t v);
  void update_rate_next(const Array &v);
  void swap();
  void compute_expectations();
  void initialize();

  double compute_elbo_term_helper() const;
  void save_state(const IDMap &m) const;

private:
  uint32_t _n;
  double _sprior;
  double _rprior;
  Array _scurr;      // current variational shape posterior 
  Array _snext;      // to help compute gradient update
  Array _rcurr;      // current variational rate posterior (global)
  Array _rnext;      // help compute gradient update
  Array _Ev;         // expected weights under variational
		      // distribution
  Array _Elogv;      // expected log weights 
  gsl_rng **_r;
};

inline void
GPArray::set_to_prior()
{
  _snext.set_elements(_sprior);
  _rnext.set_elements(_rprior);
}

inline void
GPArray::update_shape_next(const Array &sphi)
{
  assert (sphi.size() == _n);
  _snext += sphi;
}

inline void
GPArray::update_shape_next(uint32_t n, uint32_t v)
{
  _snext[n] += v;
}

inline void
GPArray::update_rate_next(const Array &v)
{
  assert (v.size() == _n);
  _rnext += v;
}

inline void
GPArray::swap()
{
  _scurr.swap(_snext);
  _rcurr.swap(_rnext);
  set_to_prior();
}

inline void
GPArray::compute_expectations()
{
  const double * const ad = _scurr.const_data();
  const double * const bd = _rcurr.const_data();
  double *vd1 = _Ev.data();
  double *vd2 = _Elogv.data();
  double a = .0, b = .0;
  for (uint32_t i = 0; i < _n; ++i) {
    make_nonzero(ad[i], bd[i], a, b);
    vd1[i] = a / b;
    vd2[i] = gsl_sf_psi(a) - log(b);
  }
}

inline void
GPArray::initialize()
{
  double *ad = _scurr.data();
  double *bd = _rcurr.data();
  for (uint32_t i = 0; i < _n; ++i) {
    ad[i] = _sprior + 0.001 * gsl_rng_uniform(*_r);
    bd[i] = _rprior + 0.1 * gsl_rng_uniform(*_r);
  }
  
  double *vd1 = _Ev.data();
  double *vd2 = _Elogv.data();
  
  double v = _rprior + _n;
  for (uint32_t i = 0; i < _n; ++i) {
    vd1[i] = ad[i] / v;
    vd2[i] = gsl_sf_psi(ad[i]) - log(v);
  }
} 

inline double
GPArray::compute_elbo_term_helper() const
{
  const double *etheta = _Ev.const_data();
  const double *elogtheta = _Elogv.const_data();
  const double * const ad = shape_curr().const_data();
  const double * const bd = rate_curr().const_data();

  double s = .0;
  double a = .0, b = .0;
  for (uint32_t n = 0; n < _n; ++n)  {
    make_nonzero(ad[n], bd[n], a, b);
    s += _sprior * log(_rprior) + (_sprior - 1) * elogtheta[n];
    s -= _rprior * etheta[n] + gsl_sf_lngamma(_sprior);
    s -= a * log(b) + (a - 1) * elogtheta[n];
    s += b * etheta[n] + gsl_sf_lngamma(a);
  }
  return s;
}

inline void
GPArray::save_state(const IDMap &m) const
{
  string fname = string("/") + name() + ".tsv";
  FILE * tf = fopen(Env::file_str(fname.c_str()).c_str(), "w");
  const double *gd = expected_v().data();
  uint32_t id = 0;
  for (uint32_t i = 0; i < _n; ++i) {
    IDMap::const_iterator idt = m.find(i);
    if (idt != m.end()) 
      id = idt->second;
    else
      id = i;
    
    fprintf(tf,"%d\t", i);
    fprintf(tf,"%d\t", id);
    fprintf(tf,"%.8f\n", gd[i]);
  }
  fclose(tf);
}

#endif
