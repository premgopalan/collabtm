#ifndef RATINGS_HH
#define RATINGS_HH

#include <string>
#include <vector>
#include <queue>
#include <map>
#include <stdint.h>
#include "matrix.hh"
#include "env.hh"
#include <string.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf.h>

using namespace std;

class Ratings {
public:
  Ratings(Env &env):
    _y(env.n,env.m), 
    _users(env.n),
    _movies(env.m),
    _env(env),
    _curr_user_seq(0), 
    _curr_movie_seq(0),
    _nratings(0),
    _likes(0) { }
  ~Ratings() { }

  int read(string s);
  
  const AdjMatrix &y() const { return _y; }
  AdjMatrix &y() { return _y; }
  
  const SparseMatrix &users() const { return _users; }
  SparseMatrix &users() { return _users; }

  const SparseMatrix &movies() const { return _movies; }
  SparseMatrix &movies() { return _movies; }
  
  uint32_t n() const;
  uint32_t m() const;
  yval_t r(uint32_t i, uint32_t j) const;
  uint32_t nratings() const { return _nratings; }
  uint32_t likes() const { return _likes; }
 
  const vector<uint32_t> *get_users(uint32_t a);
  const vector<uint32_t> *get_movies(uint32_t a);

  const IDMap &user2seq() const { return _user2seq; }
  const IDMap &seq2user() const { return _seq2user; }

  const IDMap &movie2seq() const { return _movie2seq; }
  const IDMap &seq2movie() const { return _seq2movie; }

private:
  int read_netflix_movie(string dir, uint32_t movie);
  string movies_by_user_s() const;
  bool add_movie(uint32_t id);
  bool add_user(uint32_t id);

  AdjMatrix _y;
  SparseMatrix _users;
  SparseMatrix _movies;

  Env &_env;
  IDMap _user2seq;
  IDMap _movie2seq;
  IDMap _seq2user;
  IDMap _seq2movie;
  uint32_t _curr_user_seq;
  uint32_t _curr_movie_seq;
  uint32_t _nratings;
  uint32_t _likes;
};

inline uint32_t
Ratings::n() const
{
  return _users.size();
}

inline uint32_t
Ratings::m() const
{
  return _movies.size();
}

inline bool
Ratings::add_user(uint32_t id)
{
  if (_curr_user_seq >= _env.n) {
    lerr("max users %d reached", _env.n);
    return false;
  }
  _user2seq[id] = _curr_user_seq;
  _seq2user[_curr_user_seq] = id;

  assert (!_users[_curr_user_seq]);
  std::vector<uint32_t> **v = _users.data();
  v[_curr_user_seq] = new vector<uint32_t>;
  _curr_user_seq++;
  return true;
}

inline bool
Ratings::add_movie(uint32_t id)
{
  if (_curr_movie_seq >= _env.m) {
    lerr("max movies %d reached", _env.m);
    return false;
  }
  _movie2seq[id] = _curr_movie_seq;
  _seq2movie[_curr_movie_seq] = id;

  assert (!_movies[_curr_movie_seq]);
  std::vector<uint32_t> **v = _movies.data();
  v[_curr_movie_seq] = new vector<uint32_t>;
  _curr_movie_seq++;
  return true;
}

inline yval_t
Ratings::r(uint32_t a, uint32_t b) const
{
  assert (a < _y.m() && b < _y.n());
  const yval_t **yd = _y.const_data();
  return yd[a][b];
}

inline const vector<uint32_t> *
Ratings::get_users(uint32_t a)
{
  assert (a < _movies.size());
  const vector<uint32_t> *v = _movies[a];
  return v;
}

inline const vector<uint32_t> *
Ratings::get_movies(uint32_t a)
{
  assert (a < _users.size());
  const vector<uint32_t> *v = _users[a];
  return v;
}

#endif
