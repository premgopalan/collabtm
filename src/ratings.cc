#include "ratings.hh"
#include "log.hh"

int
Ratings::read(string s)
{
  fprintf(stdout, "+ reading ratings dataset from %s\n", s.c_str());
  fflush(stdout);

  for (uint32_t i = 0; i < _env.m; ++i) {
    if (read_netflix_movie(s,i+1) < 0) {
      lerr("error adding movie %d\n", i);
      return -1;
    }
  }
  printf("\n+ done reading\n");
  return 0;
}

int
Ratings::read_netflix_movie(string dir, uint32_t movie)
{
  yval_t **yd = _y.data();

  char buf[1024];
  sprintf(buf, "%s/mv_%.7d.txt", dir.c_str(), movie);

  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }

  uint32_t mid = 0;
  if (!fscanf(f, "%d:\n", &mid)) {
    fclose(f);
    return -1;
  }
  assert (mid == movie);
  
  IDMap::iterator mt = _movie2seq.find(mid);
  if (mt == _movie2seq.end() && !add_movie(mid)) {
    fclose(f);
    return 0;
  }
  
  uint32_t m = _movie2seq[mid];
  uint32_t uid = 0, rating = 0;
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%u,%u,%*s\n", &uid, &rating, b) < 0) {
	printf("error: unexpected lines in file\n");
	fclose(f);
	exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid))
      continue;

    uint32_t n = _user2seq[uid];
    info("n = %d, m = %d, rating = %d\n", n, m, rating);
    yd[n][m] = rating >= 4 ? 1 : 0;
    _nratings++;
    _users[n]->push_back(m);
    _movies[m]->push_back(n);
    if (rating == 4 || rating == 5)
      _likes++;
  }
  printf("\r+ read %d users, %d movies, %d ratings, %d likes", 
	 _curr_user_seq, _curr_movie_seq, _nratings, _likes);
  fflush(stdout);
  fclose(f);
  return 0;
}

string
Ratings::movies_by_user_s() const
{
  ostringstream sa;
  sa << "\n[\n";
  for (uint32_t i = 0; i < _users.size(); ++i) {
    IDMap::const_iterator it = _seq2user.find(i);
    sa << it->second << ":";
    vector<uint32_t> *v = _users[i];
    if (v)  {
      for (uint32_t j = 0; j < v->size(); ++j) {
	uint32_t m = v->at(j);
	IDMap::const_iterator mt = _seq2movie.find(m);
	sa << mt->second;
	if (j < v->size() - 1)
	  sa << ", ";
      }
      sa << "\n";
    }
  }
  sa << "]";
  return sa.str();
}
