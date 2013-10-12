//
// SOURCE: http://www.cs.cornell.edu/Courses/cs578/2002fa/roceasy.c
// Precision recall curves by Prem Gopalan, pgopalan@cs.princeton.edu
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_ITEMS 1000000
#define debug 0
#define eps 1.0e-10

  /* 
     To compile on unix, try "cc -o roceasy roceasy.c -lm"

     Input to the program is a sequence of lines, one line per 
     case with the following format: 

     "TRUE_VALUE whitespace PROB_CLASS_1"

     TRUE_VALUE is the true class (usually a 0 or 1).
     PROB_CLASS_1 is the probability predicted for class 1.

     To run roc.hw3 on the results in a file called "test_preds"

          roc.hw3 < test_preds 

     The program computes the ROC Area, RMSE, and accuracy.  It also
     will compute the x,y values for the roc graph when run using the 
     option "-plot".  You'll usually want to save the x,y values in a
     file to use with a plotting program:

          roc.hw3 -plot < test_preds > roc.plot

     The ROC curve is a plot of SENSITIVITY vs. 1-SPECIFICITY as the
     prediction threshold sweeps from 0 to 1.  SENSITIVITY is also
     called True Positives, and 1-SPECIFICITY is called FALSE POSITIVES.

     A typical ROC curve for reasonably accurate predictions looks 
     like this:

              |                                   *
          S   |                         *        
          E   |                 *
          N   |           *
          S   |               
          I   |       *  
          T   |       
          I   |    *
          V   |    
          I   |  *
          T   |  
          Y   |*
              - - - - - - - - - - - - - - - - - - - 
                             1 - SPECIFICITY

     If there is no relationship between the prediction and truth, the
     ROC curve is a diagonal line with area 0.5.  If the prediction
     strongly predicts truth, the curve rises quickly and has area near
     1.0.  If the prediction strongly predicts anti-truth, the ROC area
     is less than 0.5.

     Here's a definition of SPECIFICITY, SENSITIVITY, and the other
     error measures this program computes.  (This is from Constantin's
     email.)

                           MODEL PREDICTION

                      |       1       0       |
                - - - + - - - - - - - - - - - + - - - - -
     TRUE         1   |       A       B       |    A+B
    OUTCOME           |                       |
                  0   |       C       D       |    C+D
                - - - + - - - - - - - - - - - + - - - - -
                      |      A+C     B+D      |  A+B+C+D


                1 = POSITIVE
                0 = NEGATIVE


                ACC = (A+D) /(A+B+C+D)
                PPV = A / (A+C)
                NPV = D / (B+D)
                SEN = A / (A+B)
                SPE = D / (C+D)


     WARNING!: This code has not been thoroughly tested.  If you find
               an error, please email me: caruana@cs.cornell.edu

  */

float  true[MAX_ITEMS];
float  pred[MAX_ITEMS];
double mean_true, mean_pred;
double p1;
double sse, rmse;
double pred_thresh;
int    a, b, c, d;
double freq_thresh, threshold;
double max_acc, max_acc_thresh, last_acc_thresh, acc, acc_plot;
int    freq_a, freq_b, freq_c, freq_d;
int    max_acc_a, max_acc_b, max_acc_c, max_acc_d, max_acc_count;

int arg, taken, area, plot, stats, thresh, prec_opt;
int ace, breyce, monti;
int no_item, item;
int tt, tf, ft, ff;
int total_true_0, total_true_1;
double sens, spec, tpf, fpf, tpf_prev, fpf_prev, roc_area;
double prec_area, prec_prev;
double prec;
double ace_sum, breyce_sum;

/* compute the accuracy using the threshold */

double accuracy (double threshold)
{
  int a,b,c,d,item;
  a = 0; b = 0; c = 0; d = 0;
  for (item=0; item<no_item; item++)
    if ( true[item] == 1 )
    /* true outcome = 1 */
      {
	if ( pred[item] >= threshold )
	  a++;
	else
	  b++;
      }
    else
    /* true outcome = 0 */
      {
	if ( pred[item] >= threshold )
	  c++;
	else
	  d++;
      }
  return( ((double)(a+d)) / (((double)(a+b+c+d)) + eps) );
}

/* partition is used by quicksort */

int partition (p,r)
     int p,r;
{
  int i, j;
  float x, tempf;
  
  x = pred[p];
  i = p - 1;
  j = r + 1;
  while (1)
    {
      do j--; while (!(pred[j] <= x));
      do i++; while (!(pred[i] >= x));
      if (i < j)
	{
	  tempf = pred[i];
	  pred[i] = pred[j];
	  pred[j] = tempf;
	  tempf = true[i];
	  true[i] = true[j];
	  true[j] = tempf;
	}
      else
	return(j);
    }
}

/* vanilla quicksort */

quicksort (p,r)
     int p,r;
{
  int q;
  
  if (p < r)
    {
      q = partition (p,r);
      quicksort (p,q);
      quicksort (q+1,r);
    }
}

main (argc, argv)
     int  argc;
     int  **argv;

{
  area = 1;
  plot = 0;
  stats = 1;
  arg = 1;
  prec_opt = 0;
  while ( arg < argc )
    {
      if (!strcmp((char *)argv[arg], "-p")        ||
	  !strcmp((char *)argv[arg], "-rocplot")  ||
	  !strcmp((char *)argv[arg], "-ROCplot")  ||
	  !strcmp((char *)argv[arg], "-roc_plot") ||
	  !strcmp((char *)argv[arg], "-ROC_plot") ||
	  !strcmp((char *)argv[arg], "-plot"))
      {
	   plot = 1;
	   area = 0;
	   stats = 0;
	   taken = 1;
      } else if (!strcmp((char *)argv[arg], "-prec")) {
	   //printf("precision option set\n");
	   prec_opt = 1;
	   taken = 1;
	   stats = 0;
	   area = 0;
      }
      if (!taken)
	{
	  printf("\nWarning!: Unrecognized program option %s\n", argv[arg]);
	}
      arg++;
    }
  
  no_item = 0;
  mean_true = 0.0;
  mean_pred = 0.0;
  sse = 0.0;
  while ( (scanf("%f %lf", &true[no_item], &p1)) != EOF )
    {
      /* 
         Note that unlike the rest of the code, the RMSE calculation 
         assumes the class is 0 or 1 and that the probabilities 
         probably have been correctly normalized.
      */
      pred[no_item] = p1;
      sse+= (true[no_item]-p1)*(true[no_item]-p1);
      mean_true+= true[no_item];
      mean_pred+= pred[no_item];
      no_item++;
      if ( no_item >= MAX_ITEMS )
	{
	  printf ("Aborting.  Exceeded %d items.\n", MAX_ITEMS);
	  exit(1);
	}
    }
  mean_true = mean_true / ((double) no_item);
  mean_pred = mean_pred / ((double) no_item);
  rmse      = sqrt (sse / ((double) no_item));

  if (debug)
    {
      printf("%d pats read. mean_true %6.4lf. mean_pred %6.4lf\n", no_item, mean_true, mean_pred);
      fflush(stdout);
    }

  total_true_0 = 0;
  total_true_1 = 0;
  for (item=0; item<no_item; item++)
    if ( true[item] < mean_true )
      {
	true[item] = 0;
	total_true_0++;
      }
    else
      {
	true[item] = 1;
	total_true_1++;
      }

  /* sort data by predicted value */

  quicksort (0,(no_item-1));

  /* find the prediction threshold that maximizes accuracy */

  max_acc = -9.9e10;
  max_acc_thresh = 0.0;
  last_acc_thresh = 0.0;
  max_acc_count = 1;
  for (item=0; item<(no_item-1); item++)
    {
      threshold = (pred[item] + pred[item+1]) / 2.0;
      acc = accuracy(threshold);
      if ( acc > max_acc )
	{
	  max_acc = acc;
	  max_acc_thresh = threshold;
	  last_acc_thresh = threshold;
	  max_acc_count = 1;
	}
      if ( (acc == max_acc) && (threshold != last_acc_thresh) )
	{
	  max_acc_count++;
	  last_acc_thresh = threshold;
	}
    }

  /*  find the prediction threshold such that the predicted number    */
  /*  of 0's and 1's matches the observed number of true 0's and 1's  */

  freq_thresh = (pred[total_true_0]+pred[total_true_0-1])/2.0;

  /* now update some statistics using the various thresholds */

  a = 0; 
  b = 0; 
  c = 0; 
  d = 0;
  freq_a = 0;
  freq_b = 0;
  freq_c = 0;
  freq_d = 0;
  max_acc_a = 0;
  max_acc_b = 0;
  max_acc_c = 0;
  max_acc_d = 0;
  for (item=0; item<no_item; item++)
    if ( true[item] == 1 )
    /* true outcome = 1 */
      {
	if ( pred[item] >= pred_thresh )
	  a++;
	else
	  b++;
	if ( pred[item] >= freq_thresh )
	  freq_a++;
	else
	  freq_b++;
	if ( pred[item] >= max_acc_thresh )
	  max_acc_a++;
	else
	  max_acc_b++;
      }
    else
    /* true outcome = 0 */
      {
	if ( pred[item] >= pred_thresh )
	  c++;
	else
	  d++;
	if ( pred[item] >= freq_thresh )
	freq_c++;
      else
	freq_d++;
	if ( pred[item] >= max_acc_thresh )
	max_acc_c++;
      else
	max_acc_d++;
      }

  /* now let's do the ROC cruve and area */

  tt = 0; 
  tf = total_true_1; 
  ft = 0; 
  ff = total_true_0;

  sens = ((double) tt) / ((double) (tt+tf));
  spec = ((double) ff) / ((double) (ft+ff));
  prec = 0;
  tpf = sens;
  fpf = 1.0 - spec;
  if ( plot )
       printf ("%6.4lf %6.4lf\n", fpf, tpf);
  else if ( prec_opt )
       printf ("%6.4lf %6.4lf\n", sens, (prec >= 0) ? prec : .0);
  roc_area = 0.0;
  tpf_prev = tpf;
  fpf_prev = fpf;
  prec_prev = prec;
  prec_area = 0.0;

  for (item=no_item-1; item>-1; item--)
    {
      tt+= true[item];
      tf-= true[item];
      ft+= 1 - true[item];
      ff-= 1 - true[item];
      sens = ((double) tt) / ((double) (tt+tf));
      spec = ((double) ff) / ((double) (ft+ff));
      prec = ((double) tt) / ((double) (tt+ft));
      tpf  = sens;
      fpf  = 1.0 - spec;
      if ( item > 0 )
	if ( pred[item] != pred[item-1] )
	  {
	    if ( plot )
	      printf ("%6.4lf %6.4lf\n", fpf, tpf);
	    else if ( prec_opt )
		 printf ("%6.4lf %6.4lf\n", sens, prec);	  
	    roc_area+= 0.5*(tpf+tpf_prev)*(fpf-fpf_prev);
	    prec_area+= 0.5*(tpf+tpf_prev)*fabs(prec-prec_prev);
	    tpf_prev = tpf;
	    fpf_prev = fpf;
	    prec_prev = prec;
	  }
      if ( item == 0 )
	{
	  if ( plot )
	    printf ("%6.4lf %6.4lf\n", fpf, tpf);
	  roc_area+= 0.5*(tpf+tpf_prev)*(fpf-fpf_prev);
	  prec_area+= 0.5*(tpf+tpf_prev)*fabs(prec-prec_prev);
	}
    }

  if ( stats )
    {
      printf ("ACC %8.5lf\n", accuracy(0.5));
      printf ("RMS %8.5lf\n", rmse);
      printf ("ROC %8.5lf\n", roc_area);
      printf ("AUC-PR %8.5lf\n", prec_area);
    }
}
