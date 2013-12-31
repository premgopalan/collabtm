#include "ratings.hh"
#include "log.hh"
#include <wchar.h>

int
Ratings::read(string s)
{
  fprintf(stdout, "+ reading ratings dataset from %s\n", s.c_str());
  fflush(stdout);

  if (_env.mode == Env::CREATE_TRAIN_TEST_SETS) {
    if (_env.dataset == Env::NETFLIX) {
      for (uint32_t i = 0; i < _env.ndocs; ++i) {
	if (read_netflix_movie(s,i+1) < 0) {
	  lerr("error adding movie %d\n", i);
	  return -1;
	}
      }
    } else if (_env.dataset == Env::MOVIELENS)
      read_movielens(s);
    else  if (_env.dataset == Env::MENDELEY)
      read_mendeley(s);
    else if (_env.dataset == Env::ECHONEST)
      read_echonest(s);
  } else  {
    // this ordering allows doc id == doc seq
    // a necessary condition while loading LDA fits
    // into collabtm's beta and theta
    if (_env.use_docs) {
      read_generic_docs(s);
      _movies_read = true;
    }
    if (_env.use_ratings)
      read_generic_train(s);
  }
    
  char st[1024];
  sprintf(st, "read %d users, %d movies, %d ratings", 
	  _curr_user_seq, _curr_movie_seq, _nratings);
  Env::plog("statistics", string(st));

  return 0;
}

int
Ratings::read_generic_train(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/train.tsv", dir.c_str());
  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }

  read_generic(f, NULL);
  fclose(f);
  Env::plog("training ratings", _nratings);

  _env.nusers = _curr_user_seq;
  _env.ndocs = _curr_movie_seq;

  Env::plog("nusers = ", _env.nusers);
  Env::plog("ndocs = ", _env.ndocs);
}

int
Ratings::read_generic(FILE *f, CountMap *cmap)
{
  assert(f);
  char b[128];
  uint32_t mid = 0, uid = 0, rating = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\n", &uid, &mid, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }
    IDMap::iterator mt = _movie2seq.find(mid);
    if (_movies_read && mt == _movie2seq.end())  // skip this entry
      continue;

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      //lerr("error: exceeded user limit %d, %d, %d\n",
      //uid, mid, rating);
      //fflush(stdout);
      continue;
    }
    
    if (mt == _movie2seq.end() && !add_movie(mid)) {
      printf("error: exceeded movie limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];
    
    if (rating > 0) {
      if (!cmap) {
	_nratings++;
	RatingMap *rm = _users2rating[n];
	if (_env.binary_data)
	  (*rm)[m] = 1;
	else
	  (*rm)[m] = rating;
	_users[n]->push_back(m);
	_movies[m]->push_back(n);
      } else {
	Rating r(n,m);
	assert(cmap);
	if (_env.binary_data)
	  (*cmap)[r] = 1;
	else
	  (*cmap)[r] = rating;
      }
    }
  }
  return 0;
}

int
Ratings::read_test_users(FILE *f, UserMap *bmap)
{
  assert (bmap);
  uint32_t uid = 0;
  while (!feof(f)) {
    if (fscanf(f, "%u\n", &uid) < 0) {
      printf("error: unexpected lines in file\n");
      exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    assert (it != _user2seq.end());
    uint32_t n = _user2seq[uid];

    (*bmap)[n] = true;
  }
  Env::plog("read %d test users", bmap->size());
  return 0;
}

int
Ratings::read_echonest(string dir)
{
  printf("reading echo nest dataset...\n");
  fflush(stdout);
  uint32_t mcurr = 1, scurr = 1;
  char buf[1024];
  sprintf(buf, "%s/train_triplets.txt", dir.c_str());

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  uint32_t mid = 0, uid = 0, rating = 0;
  char mids[512], uids[512];
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%s\t%s\t%u\n", uids, mids, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    StrMap::iterator uiditr = _str2id.find(uids);
    if (uiditr == _str2id.end()) {
      _str2id[uids] = scurr;
      scurr++;
    }
    uid = _str2id[uids];
    
    StrMap::iterator miditr = _str2id.find(mids);
    if (miditr == _str2id.end()) {
      _str2id[mids] = mcurr;
      mcurr++;
    }
    mid = _str2id[mids];

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    IDMap::iterator mt = _movie2seq.find(mid);
    if (mt == _movie2seq.end() && !add_movie(mid)) {
      printf("error: exceeded movie limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];

    _user2str[n] = uids;
    _movie2str[m] = mids;

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
    if (_nratings % 1000 == 0) {
      printf("\r+ read %d users, %d movies, %d ratings", 
	     _curr_user_seq, _curr_movie_seq, _nratings);
      fflush(stdout);
    }
  }
  fclose(f);
  return 0;
}

int
Ratings::read_mendeley(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/users.dat", dir.c_str());
  
  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  
  uint32_t uid = 1, rating = 0;
  char b[128];
  while (!feof(f)) {
    vector<uint32_t> mids;
    uint32_t len = 0;
    if (fscanf(f, "%u\t", &len) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    uint32_t mid = 0;
    for (uint32_t i = 0; i < len; ++i) {
      if (i == len - 1) {
	if (fscanf(f, "%u\t", &mid) < 0) {
	  printf("error: unexpected lines in file\n");
	  fclose(f);
	  exit(-1);
	}
	mids.push_back(mid);
      } else {
	if (fscanf(f, "%u", &mid) < 0) {
	  printf("error: unexpected lines in file\n");
	  fclose(f);
	  exit(-1);
	}
	mids.push_back(mid);
      }
    }
    
    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    for (uint32_t m = 0; m < mids.size(); ++m) {
      uint32_t mid = mids[m];
      IDMap::iterator mt = _movie2seq.find(mid);
      if (mt == _movie2seq.end() && !add_movie(mid)) {
	printf("error: exceeded movie limit %d, %d, %d\n",
	       uid, mid, rating);
	fflush(stdout);
	continue;
      }
      uint32_t m = _movie2seq[mid];
      uint32_t n = _user2seq[uid];

      yval_t rating = 1.0;
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
    uid++;
    if (_nratings % 1000 == 0) {
      printf("\r+ read %d users, %d movies, %d ratings", 
	     _curr_user_seq, _curr_movie_seq, _nratings);
      fflush(stdout);
    }
  }
  fclose(f);
  return 0;
}

int
Ratings::read_generic_docs(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/mult.dat", dir.c_str());
  
  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  
  uint32_t maxwid = 0;
  uint32_t docid = 0, docseq = 0;
  char b[128];
  while (!feof(f)) {
    vector<uint32_t> mids;
    uint32_t len = 0;
    if (fscanf(f, "%u\t", &len) < 0) {
      lerr("error: unexpected lines in file\n");
      fclose(f);
      lerr("docseq = %d, docid = %d, maxwid = %d\n", docseq, docid, maxwid);
      _env.ndocs = _curr_movie_seq;
      Env::plog("ndocs loaded (limited by nusers)", _env.ndocs);
      return 0;
    }

    IDMap::iterator mt = _movie2seq.find(docid);
    if (mt == _movie2seq.end() && !add_movie(docid)) {
      Env::plog("ndocs loaded (limited by ndocs)", _curr_movie_seq);
      lerr("warning: exceeded movie limit %d\n", docid);
      lerr("docseq = %d, docid = %d, maxwid = %d\n", docseq, docid, maxwid);
      fclose(f);
      return 0;
    }
    mt = _movie2seq.find(docid);
    docseq = mt->second;
    debug("found docid %d docseq %d\n", docid, docseq);
    
    uint32_t wid = 1, wc = 0;
    for (uint32_t i = 0; i < len; ++i) {
      if (i == len - 1) {
	if (fscanf(f, "%u:%u\n", &wid, &wc) < 0) {
	  printf("error: unexpected lines in file\n");
	  fclose(f);
	  exit(-1);
	}
      } else {
	if (fscanf(f, "%u:%u\t", &wid, &wc) < 0) {
	  printf("error: unexpected lines in file\n");
	  fclose(f);
	  exit(-1);
	}
      }
      if (!_docs2words[docseq]) {
	WordVec **wm = _docs2words.data();
	wm[docseq] = new WordVec;
      } 
      WordVec *wm = _docs2words[docseq];
      wm->push_back(WordCount(wid, wc));

      if (wid > maxwid) {
	maxwid = wid;
      }
    }
    docid++;
  }
  printf("docseq = %d, docid = %d, maxwid = %d\n", docseq, docid-1, maxwid);
  fflush(stdout);
  fclose(f);
  return 0;
}

int
Ratings::read_netflix_movie(string dir, uint32_t movie)
{
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

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
  }
  fclose(f);
  printf("\r+ read %d users, %d movies, %d ratings", 
	 _curr_user_seq, _curr_movie_seq, _nratings);
  fflush(stdout);
  return 0;
}

int
Ratings::read_movielens(string dir)
{
  char buf[1024];
  sprintf(buf, "%s/ml-1m_train.tsv", dir.c_str());

  info("reading from %s\n", buf);

  FILE *f = fopen(buf, "r");
  if (!f) {
    fprintf(stderr, "error: cannot open file %s:%s", buf, strerror(errno));
    fclose(f);
    exit(-1);
  }
  
  uint32_t mid = 0, uid = 0, rating = 0;
  char b[128];
  while (!feof(f)) {
    if (fscanf(f, "%u\t%u\t%u\n", &uid, &mid, &rating) < 0) {
      printf("error: unexpected lines in file\n");
      fclose(f);
      exit(-1);
    }

    IDMap::iterator it = _user2seq.find(uid);
    if (it == _user2seq.end() && !add_user(uid)) {
      printf("error: exceeded user limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    IDMap::iterator mt = _movie2seq.find(mid);
    if (mt == _movie2seq.end() && !add_movie(mid)) {
      printf("error: exceeded movie limit %d, %d, %d\n",
	     uid, mid, rating);
      fflush(stdout);
      continue;
    }
    
    uint32_t m = _movie2seq[mid];
    uint32_t n = _user2seq[uid];

    if (rating > 0) {
      _nratings++;
      RatingMap *rm = _users2rating[n];
      (*rm)[m] = rating;
      _users[n]->push_back(m);
      _movies[m]->push_back(n);
      _ratings.push_back(Rating(n,m));
    }
  }
  fclose(f);
  return 0;
}


void
Ratings::load_movies_metadata(string s)
{
  if (_env.dataset == Env::MOVIELENS)
    read_movielens_metadata(s);
  else if (_env.dataset == Env::NETFLIX)
    read_netflix_metadata(s);
  else if (_env.dataset == Env::MENDELEY)
    read_mendeley_metadata(s);
}

int
Ratings::read_movielens_metadata(string dir)
{
  uint32_t n = 0;
  char buf[1024];
  sprintf(buf, "%s/movies.tsv", dir.c_str());
  FILE *f = fopen(buf, "r");
  assert(f);
  uint32_t id;
  char name[4096];
  char type[4096];
  char *line = (char *)malloc(4096);
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    uint32_t k = 0;
    char *p = line;
    const char r[3] = "#";
    do {
      char *q = NULL;
      char *d = strtok_r(p, r, &q);
      if (q == p)
	break;
      if (k == 0) {
	id = atoi(d);
	id = _movie2seq[id];
      } else if (k == 1) {
	strcpy(name, d);
	_movie_names[id] = name;
	lerr("%d -> %s", id, name);
      } else if (k == 2) {
	strcpy(type, d);
	_movie_types[id] = type;
	lerr("%d -> %s", id, type);
      }
      p = q;
      k++;
    } while (p != NULL);
    n++;
    debug("read %d lines\n", n);
    memset(line, 0, 4096);
  }
  free(line);
  return 0;
}

int
Ratings::read_netflix_metadata(string dir)
{
  uint32_t n = 0;
  char buf[1024];
  sprintf(buf, "%s/movie_titles.txt", dir.c_str());
  FILE *f = fopen(buf, "r");
  assert(f);
  uint32_t id, year;
  char name[4096];
  char *line = (char *)malloc(4096);
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    uint32_t k = 0;
    char *p = line;
    const char r[3] = ",";
    do {
      char *q = NULL;
      char *d = strtok_r(p, r, &q);
      if (q == p)
	break;
      if (k == 0) {
	id = atoi(d);
	lerr("%d: ", id);
	id = _movie2seq[id];
	lerr("%d -> ", id);
      } else if (k == 1) {
	year  = atoi(d); // skip
	lerr("%d", year);
      } else if (k == 2) {
	strcpy(name, d);
	_movie_names[id] = name;
	_movie_types[id] = "";
	lerr("%d -> %s", id, name);
      }
      p = q;
      k++;
    } while (p != NULL);
    n++;
    lerr("read %d lines\n", n);
    memset(line, 0, 4096);
  }
  free(line);
  return 0;
}

int
Ratings::read_mendeley_metadata(string dir)
{
  uint32_t n = 0;
  char buf[1024];
  sprintf(buf, "%s/titles.dat", dir.c_str());
  FILE *f = fopen(buf, "r");
  assert(f);
  char name[4096];
  char *line = (char *)malloc(4096);
  uint32_t id = 0;
  while (!feof(f)) {
    if (fgets(line, 4096, f) == NULL)
      break;
    strcpy(name, line);
    uint32_t seq = _movie2seq[id];
    _movie_names[seq] = name;
    id++;
  }
  lerr("read %d lines\n", n);
  free(line);
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
