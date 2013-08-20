
// void
// GAPRec::set_precision_user_sample()
// {
//   UserMap usermap;
//   do {
//     uint32_t user = gsl_rng_uniform_int(_r, _n);
//     const vector<uint32_t> *movies = _ratings.get_movies(user);

//     if (!movies || movies->size() < 10)
//       continue;
//     usermap[user] = true;
    
//     uint32_t msize = movies->size();
//     uArray liked_movies(msize);
//     for (uint32_t j = 0, k = 0; j < msize; ++j)
//       liked_movies[k++] = movies->at(j);
//     gsl_ran_shuffle(_r, (void *)liked_movies.data(), deg, sizeof(uint32_t));
    
//     for (uint32_t k = 0; k < msize * 0.3; k++) {


//     Rating r;
//     get_random_rating(r);
//     _test_ratings.push_back(r);
//     _test_map[r] = true;
//     n++;
//   }
//   FILE *hef = fopen(Env::file_str("/test-ratings.txt").c_str(), "w");  
//   fprintf(hef, "%s\n", ratingslist_s(_test_map).c_str());
//   fclose(hef);
    

//   } while (

// }


// void
// GAPRec::save_model()
// {
//   FILE *tf = fopen(Env::file_str("/theta.txt").c_str(), "w");  
//   double **gd = _Etheta.data();
//   double s = .0;
//   for (uint32_t i = 0; i < _n; ++i) {
//     const IDMap &m = _network.seq2id();
//     IDMap::const_iterator idt = m.find(i);
//     if (idt != m.end()) {
//       fprintf(tf,"%d\t", i);
//       debug("looking up i %d\n", i);
//       fprintf(tf,"%d\t", (*idt).second);
//       for (uint32_t k = 0; k < _k; ++k) {
// 	if (k == _k - 1)
// 	  fprintf(tf,"%.5f\n", gd[i][k]);
// 	else
// 	  fprintf(tf,"%.5f\t", gd[i][k]);
//       }
//     }
//   }
//   fclose(tf);
// }
