/* BSP heuristic - 0/1 loss + errormargin + geometric margin */
/* Copyright Usman Roshan 2015 */

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

#define DEBUG 0
#define OUTPUTLEVEL 0
#define OUTPUTTIME 0

short __TEMPFLAG__ = 0;

/*define variables*/
short *test_sign;
uint8_t *sign, do_random=0;
double *projection, *test_projection, C = 1;
double *w, w0, w_length, old_w_length, bestw0, oldw0, oldval, tempval, newval, bestval, *column;
double *initial_w, initial_w0, *w_ils, *w_rr, w0_ils, w0_rr;
double  **data, **testdata, plus=0, minus=0, dbl, origplus=0, origminus=0;
double* allw0;
short *allzeros, *allsame;
int nfeatures, *len;
double below, above, z, below_prime, thresh, thresh2;
int *rows_j, *dup, localit=0, localit_=0, *column_numbers, *row_numbers, *column_change;
int *label, *origlabel, *test_label, redocolumn=0, startm, endm, bestm, localitthresh=1, localitthresh_=10;
int *uniq_proj_ind, uniq_proj_ind_n, extraiter, plus_=0, minus_=0;
int ils = 0, *featuremap, *rowmap, jobs, nprocs, num_iters;
int8_t **newtrain, **newtest;

struct timeval starttime, endtime; float seconds, sortseconds=0;
struct timeval starttime2, endtime2;
struct timeval starttime3, endtime3;

//objective is to maximize margin - error that peaks at 0
double errormargin, margin, error, errorplus, errorminus, objective;
double bestmargin, besterror, bestobjective, prevbestobjective;
double stopthresh = .001, w_inc = .01, initialw = 1;
double geomargin, abovedist, belowdist, aboverow, belowrow;

int rows, cols, testrows, origrows, origcols;

int compare (const void * a, const void * b)
{
  double fa = *(double*) a;
  double fb = *(double*) b;
  return (fa > fb) - (fa < fb);
}

void fisher_yates(int* array, int l){
  int i, j; int temp;
  for(i = l; i > 0; i--){
    j = rand() % (i + 1);
    if(i == j) continue;
    temp = array[j];
    array[j] = array[i];
    array[i] = temp;
  }
}

void obtain_unique_projection_values(){

  /*now obtain unique projection values*/
  int i, k;
  double tempval = projection[0];
  uniq_proj_ind[0] = 0;
  uniq_proj_ind_n = 1;
  for(k = 1; k < rows; k++) {
//    if(projection[k] - tempval > 1) {
    if(projection[k] != tempval) {
      uniq_proj_ind[uniq_proj_ind_n++] = k;
      tempval = projection[k];
    }
  }
  uniq_proj_ind[uniq_proj_ind_n] = rows;
  uniq_proj_ind_n++;

if(DEBUG > 1) {
  printf("Uniq projection indices\n");
  for(i = 0; i < uniq_proj_ind_n; i++)
    printf("%d ", uniq_proj_ind[i]);
  printf("\n");
}

}

void obtain_projection(short flag){

  int i, j;

  /*length of w*/
  w_length = 0;
  for(j = 0; j < cols; j++) { w_length += w[featuremap[j]]*w[featuremap[j]]; }
  w_length = sqrt(w_length);

  if(flag){
    for(i = 0; i < rows; i++) {
      projection[i] = 0;
      for(j = 0; j < cols; j++) {
        projection[i] += data[rowmap[i]][featuremap[j]] * w[featuremap[j]];
      }
      projection[i] /= w_length;
    }
  }
  else {
    for(i = 0; i < rows; i++) {
      projection[i] = 0;
      for(j = 0; j < cols; j++) {
        projection[i] += data[i][featuremap[j]] * w[featuremap[j]];
      }
      projection[i] /= w_length;
    }
  }
}

void obtain_projection_parallel(int jobs, short flag){

  int id = omp_get_thread_num();
  int i, j, stop; float test_proj;

  if((id+1)*jobs > rows) stop=rows;
  else stop = (id+1)*jobs;

//  printf("jobs=%d start=%d stop=%d\n", jobs, id*jobs, stop); fflush(stdout);

  for(i = id*jobs; i < stop; i++) {
    projection[i] = 0;
    for(j = 0; j < cols; j++) {
      if(!flag)
        projection[i] += data[i][j] * w[j];
      else
        projection[i] += data[i][j] * w[j];
//        projection[i] += data[rowmap[i]][j] * w[j];
//      projection[i] += data[i][featuremap[j]] * w[featuremap[j]];
    }
    projection[i] /= w_length;
  }
}

void sort_projection(){

  double temp; int i, j, k;
  for(i = 1; i < rows; i++){
    j = i;
    while (j > 0 && projection[j-1] > projection[j]) {
      temp = projection[j];
      projection[j] = projection[j-1];
      projection[j-1] = temp;
      temp = label[j];
      label[j] = label[j-1];
      label[j-1] = temp;
      temp = rowmap[j];
      rowmap[j] = rowmap[j-1];
      rowmap[j-1] = temp;
      j--;
    }
  }

if(DEBUG > 1) {
  printf("Projection\n");
  for(k = 0; k < rows; k++)
    printf("%f ", projection[k]);
  printf("\n");
  printf("Labels\n");
  for(k = 0; k < rows; k++)
    printf("%d ", label[k]);
  printf("\n");
}
}

void sort_projection_fast(){

  double temp; int j;
  for(j = 1; j < rows; j++){
    if(projection[j-1] > projection[j]) {
      temp = projection[j];
      projection[j] = projection[j-1];
      projection[j-1] = temp;
      temp = label[j];
      label[j] = label[j-1];
      label[j-1] = temp;
      temp = rowmap[j];
      rowmap[j] = rowmap[j-1];
      rowmap[j-1] = temp;
    }
  }
}

void initialize_w() {
  /*make random plane*/
  int j;
  for(j = 0; j < cols; j++) {
    if(allzeros[featuremap[j]] || allsame[featuremap[j]]) {
      w[featuremap[j]]=0;
    }
    else {
      double rd = rand(); rd = rd / (double)RAND_MAX; rd *= 2*initialw; rd -= initialw;
      w[featuremap[j]] = rd;
    }
    initial_w[featuremap[j]] = w[featuremap[j]];
  }
}

void update_wlen_and_projection(int j, int jobs){

//  int id = omp_get_thread_num();
  int k;
//  int i, stop;

/*update length of w - done in main code so that w_length is local variable for each thread */
  w_length = w_length * w_length;
  w_length -= oldval * oldval;
  w_length += newval * newval;
  w_length = sqrt(w_length);

/*take out old value from dot product and add new value - old serial version */
  for(k = 0; k < rows; k++){
    projection[k] *= old_w_length;
    projection[k] -= data[rowmap[k]][j]*oldval;
    projection[k] += data[rowmap[k]][j]*newval;
    projection[k] /= w_length;
  }
/*
  if((id+1)*jobs > rows) stop=rows;
  else stop = (id+1)*jobs;

//  printf("jobs=%d start=%d stop=%d ind=%d\n", jobs, id*jobs, stop, ind); fflush(stdout);

  for(i = id*jobs; i < stop; i++) {
    projection[i] *= old_w_length;
    projection[i] -= data[rowmap[i]][j]*oldval;
    projection[i] += data[rowmap[i]][j]*newval;
    projection[i] /= w_length;
  }
*/
}

void scan_initial_projection(int startval, short flag){

  int m, k;
  w0 = -1 * (projection[uniq_proj_ind[startval]] + projection[uniq_proj_ind[startval+1]])/2;
  error = 0; errormargin = 0; below = 0; above = 0;
  errorplus = 0, errorminus = 0;
  geomargin = FLT_MAX; abovedist = FLT_MAX; belowdist = FLT_MAX; belowrow = -1; aboverow = -1;
  for(m = 0; m < uniq_proj_ind_n - 1; m++) {
    for(k = uniq_proj_ind[m]; k < uniq_proj_ind[m+1]; k++){
      if(label[k]*(projection[k]+w0) == 0) {
        if(label[k] == 1) { errorplus++; below++; }
      }
      else if(label[k]*(projection[k]+w0) < 0) {
        if(label[k] == 1) errorplus++;
        else              errorminus++;
      }
    }
    }
  error = (errorplus + errorminus)/(plus + minus);

/*initialize bestobjective*/
  objective = -1 * error;

  if(!flag){
    bestobjective = objective; bestmargin = margin; besterror = error; bestw0 = w0; bestm = 0;
  }
  else {
    if(objective >= bestobjective) {
      bestobjective = objective;
      besterror = error;
      bestmargin = margin;
      bestw0 = w0;
      bestval = newval;
      bestm = startm;
    }
  }

if(DEBUG > 0){
  printf("Initial values: objective=%f (%f), error=%f (%f), margin=%f (%f), geomargin=%f, errormargin=%f, errorplus=%.1f, errorminus=%.1f, w0=%f\n", objective, bestobjective, error, besterror, margin, bestmargin, geomargin, errormargin, errorplus, errorminus, w0);
  fflush(stdout);
}

}

void get_random_w0(int startval, int endval){

  w0 = -1 * (projection[uniq_proj_ind[0]] + projection[uniq_proj_ind[1]])/2;
  allw0[0] = w0;

  int m, k;
  for(m = startval; m < endval; m++) {
    oldw0 = w0;
    z = ((projection[uniq_proj_ind[m]] + projection[uniq_proj_ind[m+1]])/2) - (-1*w0);
if(DEBUG > 1)
    printf("z=%.12f\n", z);
    w0 -= z;
    allw0[m] = w0;
  }

//  for(m = 0; m < endval; m++) { printf("%f ", allw0[m]); }
  double rd = rand(); rd = rd / (double)RAND_MAX;
  rd *= (endval);
  rd = (int) floor(rd);
  w0 = allw0[(int) rd];

}

void get_random_w_and_w0(){

  int j, i, j_, k;

  /*initialize*/
  localit=0;

  for(i=0;i<rows;i++) label[i]=origlabel[rowmap[i]];

  initialize_w();


  if(DEBUG > 0) {
    printf("Initial w: = ");
    for(j = 0; j < cols; j++) printf("%.10f ", w[featuremap[j]]);
    printf("\n");
  }

  /*obtain initial dot products*/
  obtain_projection(1);

  sort_projection();
  obtain_unique_projection_values();
  scan_initial_projection(0, 0);
  get_random_w0(1, uniq_proj_ind_n - 2);

}

void get_best_w0(int startval, int endval){

  int m, k;
  for(m = startval; m < endval; m++) {
    oldw0 = w0;
    z = ((projection[uniq_proj_ind[m]] + projection[uniq_proj_ind[m+1]])/2) - (-1*w0);
if(DEBUG > 1)
    printf("z=%.12f\n", z);
    w0 -= z;
    for(k = uniq_proj_ind[m]; k < uniq_proj_ind[m+1]; k++){
      if(label[k]*(projection[k]+w0) == 0) {
        if(label[k] == 1) {
          errorplus++;
        }
      }
      else if(label[k]*(projection[k]+w0) > 0) {
        if(label[k] == 1) errorplus--; else errorminus--;
      }
      else if(label[k]*(projection[k]+w0) < 0) {
        if(label[k] == 1) errorplus++; else errorminus++;
      }
    }

    error = (errorplus + errorminus)/(plus + minus);
    objective = -1 * error;

    if(__TEMPFLAG__) { printf("w0=%f obj=%f\n", w0, objective); }

    if(objective >= bestobjective) {
      bestobjective = objective;
      besterror = error;
      bestmargin = margin;
      bestw0 = w0;
      bestm = m;
      bestval = newval;
    }

if(DEBUG > 0)
    printf("Update values: objective=%f (%f), error=%f (%f), margin=%f (%f), geomargin=%f, errormargin=%f, errorplus=%.1f, errorminus=%.1f, w0=%f\n", objective, bestobjective, error, besterror, margin, bestmargin, geomargin, errormargin, errorplus, errorminus, w0);
  }
}

void get_best_w_and_w0(){

  int j, i, j_, k;

  /*initialize*/
  localit=0;

  for(i=0;i<rows;i++) label[i]=origlabel[rowmap[i]];

  if(ils == 0)
    initialize_w();

  /*or read it from file - disabled for now*/
/*  if(argc > 13) {
      FILE* f2=fopen(argv[13], "rt");
      for(i=0;i<cols;i++) { fscanf(f2, "%lf", &dbl); w[i] = dbl; }
    } */

if(DEBUG > 0) {
  printf("Initial w: = ");
  for(j = 0; j < cols; j++) printf("%.10f ", w[featuremap[j]]);
  printf("\n");
}

  /*obtain initial dot products*/
  obtain_projection(1);

  sort_projection();
  obtain_unique_projection_values();
  scan_initial_projection(0, 0);
  get_best_w0(1, uniq_proj_ind_n - 2);

/********** Main loop *********/

  prevbestobjective = (double) -FLT_MAX;
  assert(prevbestobjective < bestobjective);

  while(bestobjective - prevbestobjective > stopthresh && localit < localitthresh) {

//  prevbestobjective = bestobjective;

//if(OUTPUTLEVEL > 0){ printf("Starting iteration %d...\n", localit); fflush(stdout); }

    //permute columns
    for(j = 0; j < cols; j++){ column_numbers[j] = featuremap[j]; }
    fisher_yates(column_numbers, cols-1);

    localit_=0; redocolumn=0;
    plus_ = 0; minus_ = 0;
    for(j = 0; j < cols; j++) rows_j[j] = 2;

    for(j_ = 0; j_ < cols; j_++){

//      if(redocolumn && bestobjective - prevbestobjective > stopthresh && localit <= localitthresh ) {
      if(redocolumn && bestobjective - prevbestobjective > stopthresh && localit < localitthresh && localit_ < localitthresh_) {
        j_--;
        j = column_numbers[j_];
//        if(plus_)  column[0] = w[j] + w_inc;
//        if(minus_) column[0] = w[j] - w_inc;
//        if(plus_)  { column[0]=w[j]+w_inc/10000; column[1]=w[j]+w_inc/100; column[2]=w[j]+w_inc; column[3]=w[j]+100*w_inc; column[4]=w[j]+10000*w_inc; column[5]=w[j]+1000000*w_inc; }
//        if(minus_) { column[0]=w[j]-w_inc/10000; column[1]=w[j]-w_inc/100; column[2]=w[j]-w_inc; column[3]=w[j]-100*w_inc; column[4]=w[j]-10000*w_inc; column[5]=w[j]-1000000*w_inc; }
//        if(plus_)  { column[0]=w[j]+w_inc; column[1]=w[j]+w_inc/100; column[2]=w[j]+w_inc/10000; column[3]=w[j]+100*w_inc; column[4]=w[j]+10000*w_inc; column[5]=w[j]+1000000*w_inc; }
//        if(minus_) { column[0]=w[j]-w_inc; column[1]=w[j]-w_inc/100; column[2]=w[j]-w_inc/10000; column[3]=w[j]-100*w_inc; column[4]=w[j]-10000*w_inc; column[5]=w[j]-1000000*w_inc; }
//        if(plus_)  { column[0]=w[j]+w_inc; column[1]=w[j]+w_inc/100; column[2]=w[j]+100*w_inc; column[3]=w[j]+10000*w_inc; column[4]=w[j]+1000000*w_inc; column[5]=w[j]+100000000*w_inc; }
//        if(minus_) { column[0]=w[j]-w_inc; column[1]=w[j]-w_inc/100; column[2]=w[j]-100*w_inc; column[3]=w[j]-10000*w_inc; column[4]=w[j]-1000000*w_inc; column[5]=w[j]-100000000*w_inc; }
        if(plus_)  { column[0]=w[j]+w_inc; column[1]=w[j]+100*w_inc; column[2]=w[j]+10000*w_inc; column[3]=w[j]+1000000*w_inc; column[4]=w[j]+100000000*w_inc; column[5]=w[j]+10000000000*w_inc; column[6]=w[j]+1000000000000*w_inc; }
        if(minus_) { column[0]=w[j]-w_inc; column[1]=w[j]-100*w_inc; column[2]=w[j]-10000*w_inc; column[3]=w[j]-1000000*w_inc; column[4]=w[j]-100000000*w_inc; column[5]=w[j]-10000000000*w_inc; column[6]=w[j]-1000000000000*w_inc; }
        assert( (plus_ == 1 && minus_ == 0) || (plus_ == 0 && minus_ == 1) );
//        rows_j[j] = 1;
        rows_j[j] = 7;
        localit_++;
      }
      else {
        localit_=0;
        j = column_numbers[j_];
        column[0] = w[j];
        column[1] = w[j] + w_inc;
        column[2] = w[j] - w_inc;
/*        column[3] = w[j] + w_inc/100;
        column[4] = w[j] - w_inc/100;
        column[5] = w[j] + w_inc/10000;
        column[6] = w[j] - w_inc/10000;
        column[7] = w[j] + 100*w_inc;
        column[8] = w[j] - 100*w_inc;
        column[9] = w[j] + 10000*w_inc;
        column[10] = w[j] - 10000*w_inc;
        column[11] = w[j] + 1000000*w_inc;
        column[12] = w[j] - 1000000*w_inc; */
/*        column[3] = w[j] + w_inc/100;
        column[4] = w[j] - w_inc/100;
        column[5] = w[j] + 100*w_inc;
        column[6] = w[j] - 100*w_inc;
        column[7] = w[j] + 10000*w_inc;
        column[8] = w[j] - 10000*w_inc;
        column[9] = w[j] + 1000000*w_inc;
        column[10] = w[j] - 1000000*w_inc;
        column[11] = w[j] + 100000000*w_inc;
        column[12] = w[j] - 100000000*w_inc; */
        column[3] = w[j] + 100*w_inc;
        column[4] = w[j] - 100*w_inc;
        column[5] = w[j] + 10000*w_inc;
        column[6] = w[j] - 10000*w_inc;
        column[7] = w[j] + 1000000*w_inc;
        column[8] = w[j] - 1000000*w_inc;
        column[9] = w[j] + 100000000*w_inc;
        column[10] = w[j] - 100000000*w_inc;
        column[11] = w[j] + 10000000000*w_inc;
        column[12] = w[j] - 10000000000*w_inc;
        column[13] = w[j] + 1000000000000*w_inc;
        column[14] = w[j] - 1000000000000*w_inc;
/*	column[0] = w[j];
        column[1] = w[j] + w_inc/10000;
        column[2] = w[j] + w_inc/100;
        column[3] = w[j] + w_inc;
        column[4] = w[j] + 100*w_inc;
        column[5] = w[j] + 10000*w_inc;
        column[6] = w[j] + 1000000*w_inc;
        column[7] = w[j];
        column[8] = w[j] - w_inc/10000;
        column[9] = w[j] - w_inc/100;
        column[10] = w[j] - w_inc;
        column[11] = w[j] - 100*w_inc;
        column[12] = w[j] - 10000*w_inc;
        column[13] = w[j] - 1000000*w_inc; */
        rows_j[j] = 15;
        redocolumn=0;
        plus_=0;
        minus_=0;
      }

      if(allzeros[j] || allsame[j]) { continue; }

      prevbestobjective = bestobjective;

if(DEBUG > 1) {
  printf("Doing column %d\n", j);
  fflush(stdout);
}

      oldval = w[j];
      old_w_length = w_length;
      bestval = oldval;

if(DEBUG > 1){
  printf("Checking values in column %d... ", j);
  for(k = 0; k < rows_j[j]; k++)
    printf("%f ", column[k]);
  printf("\n");
}

      /*iterate through candidate w[j] values*/
      for(i = 0; i < rows_j[j]; i += 1 ){

        newval = column[i];

if(DEBUG > 1)
    printf("oldval=%f, newval=%f, bestval=%f\n", oldval, newval, bestval);

/*        w_length = w_length * w_length;
        w_length -= oldval * oldval;
        w_length += newval * newval;
        w_length = sqrt(w_length);
#pragma omp parallel num_threads(nprocs)
        update_wlen_and_projection(j, jobs); */

        update_wlen_and_projection(j, jobs);

        if(i==0) {
	sort_projection();
        obtain_unique_projection_values();
	}
	else{
	sort_projection_fast();
	obtain_unique_projection_values();
	}

if(DEBUG > 1) {
    printf("w_length=%f\n", w_length);
}

if(DEBUG > 1){
    printf("Current w = ");
    for(k = 0; k < cols; k++)
      if(k==j) printf("%f ", newval);
      else     printf("%f ", w[k]);
    printf("\n");
}

//        startm = 0; endm = uniq_proj_ind_n - 1;
        startm = bestm - 10; endm = bestm + 10;
//        startm = bestm - 5; endm = bestm + 5;
//        startm = bestm - 2; endm = bestm + 2;
//        startm = bestm - 1; endm = bestm + 1;
        if(startm < 0) startm = 0;
        if(endm > uniq_proj_ind_n - 1) endm = uniq_proj_ind_n - 1;

        scan_initial_projection(startm, 1);
        get_best_w0(startm+1, endm-1);

       //Do next number in the column
        oldval = newval;
        old_w_length = w_length;
      }

      //if new best value is found
      if(bestval != w[j]) {
//        printf("column number=%d, bestval=%f\n", j, bestval);
        column_change[j]++;
        redocolumn = 1;
        if(plus_ == 0 && minus_ == 0) {
//          if(bestval == column[7] || bestval == column[8] || bestval == column[9] || bestval == column[10] || bestval == column[11] || bestval == column[12]) { plus_=1; }
//          if(bestval == column[8] || bestval == column[9] || bestval == column[10] || bestval == column[11] || bestval == column[12] || bestval == column[13]) { plus_=1; }
//          else if(bestval == column[1] || bestval == column[2] || bestval == column[3] || bestval == column[4] || bestval == column[5] || bestval == column[6]) { minus_=1; }
          if(bestval == column[1] || bestval == column[3] || bestval == column[5] || bestval == column[7] || bestval == column[9] || bestval == column[11] || bestval == column[13] ) { plus_=1; }
          else if(bestval == column[2] || bestval == column[4] || bestval == column[6] || bestval == column[8] || bestval == column[10] || bestval == column[12] ||  bestval == column[14] ) { minus_=1; }
        }
      } //otherwise move to next column
      else {
        redocolumn = 0; plus_ = 0; minus_ = 0;
      }

      w[j] = bestval;
      newval = bestval;
      w0 = bestw0;

      /*length of w*/
      w_length = 0;
      for(k = 0; k < cols; k++) { w_length += w[featuremap[k]]*w[featuremap[k]]; }
      w_length = sqrt(w_length);

//#pragma omp parallel num_threads(nprocs)
//      update_wlen_and_projection(j, jobs);

      /* take out old value from dot product and add new value */
      for(k = 0; k < rows; k++){
        projection[k] *= old_w_length;
        projection[k] -= data[rowmap[k]][j]*oldval;
        projection[k] += data[rowmap[k]][j]*w[j];
        projection[k] /= w_length;
      }

  //End of doing one entire column

    }

if(OUTPUTLEVEL > 0){
  printf("Bestobjective=%.12f, besterror=%f, bestmargin=%.12f\n", bestobjective, besterror, bestmargin);
  fflush(stdout);
}

    localit++;
//End of doing one iteration through all columns

  }
}

void get_full_error(short flag){

  int m, k;
  error = 0; errormargin = 0; below = 0; above = 0;
  errorplus = 0, errorminus = 0;
  geomargin = FLT_MAX; abovedist = FLT_MAX; belowdist = FLT_MAX; belowrow = -1; aboverow = -1;
  for(m = 0; m < uniq_proj_ind_n - 1; m++) {
    for(k = uniq_proj_ind[m]; k < uniq_proj_ind[m+1]; k++){
      if(label[k]*(projection[k]+w0) == 0) {
        if(label[k] == 1) { errorplus++; below++; }
      }
      else if(label[k]*(projection[k]+w0) < 0) {
        if(label[k] == 1) errorplus++;
        else              errorminus++;
      }
    }
  }
  margin = errormargin + geomargin;
  error = (errorplus + errorminus)/(origplus + origminus);

/*initialize bestobjective*/
  objective = -1 * error;

  if(!flag){
    bestobjective = objective; bestw0 = w0; bestm = 0;
  }
  else {
    if(objective >= bestobjective) {
      bestobjective = objective;
      bestw0 = w0;
      bestval = newval;
      bestm = startm;
    }
  }

if(DEBUG > 0){
  printf("Initial values: objective=%f (%f), error=%f (%f), margin=%f (%f), geomargin=%f, errormargin=%f, errorplus=%.1f, errorminus=%.1f, w0=%f\n", objective, bestobjective, error, besterror, margin, bestmargin, geomargin, errormargin, errorplus, errorminus, w0);
  fflush(stdout);
}

}

float cross_val_score(){

  int i,j,k,n1=0,n2=0,trials=10;
  float err=0,m1=0,m2=0,avgerr=0;

  for(k=0; k<trials; k++){
    m1=0;m2=0;n1=0;n2=0;err=0;

    for(j = 0; j < rows; j++){ row_numbers[j] = j; }
    fisher_yates(row_numbers, rows-1);

    for(i = 0; i < .9*rows; i++)
      if(origlabel[row_numbers[i]] == 1) { m1 += sign[row_numbers[i]]; n1++; }
      else			       { m2 += sign[row_numbers[i]]; n2++; }
    m1 /= n1; m2 /= n2;

    for(i = .9*rows; i < rows; i++)
      if(fabs(sign[row_numbers[i]]-m1) < fabs(sign[row_numbers[i]]-m2)) {
        if(origlabel[row_numbers[i]] == -1) err++;
      }
      else {
        if(origlabel[row_numbers[i]] == 1) err++;
      }
//    printf("m1=%f m2=%f error=%f\n", m1,m2,err);
    avgerr += (float)err/(.1*rows);
  }
  avgerr /= trials;
//  printf("avgerror=%f\n", avgerr);
  return avgerr;
}

void standardize(double** data, double** testdata, int rows, int cols, int testrows){

  int i, j; double len=0;

  for(j=0; j<cols; j++){
    len=0;
    for(i=0; i<rows; i++)
      len += data[i][j]*data[i][j];
    len=sqrt(len);
    if(len != 0){
      for(i=0; i<rows; i++)
        data[i][j] /= len;
      for(i=0; i<testrows; i++)
        testdata[i][j] /= len;
    }
  }
}

void getzeroandsame(double** data, int rows, int cols){

  int i, j;

  for(j=0; j<cols; j++){
    for(i=0; i<rows; i++) {
      if(data[i][j] != 0) { allzeros[j] = 0; }
      if(i>0 && data[i][j] != data[i-1][j]) { allsame[j]=0; }
    }
  }
}

int main(int argc, char* argv[]) {

  /*read parameters*/
  rows = atoi(argv[1]);
  cols = atoi(argv[2]);
  testrows = atoi(argv[4]);
  origrows = rows;
  origcols = cols;

  // local variables
  int i, j, k, size, feat_count=0, *knn, l;
  double rd, test_proj, d, err;
  float m1, m2, s1, s2, stn, overall_stn=0, avg_stn, *mind;
  float pred = 0, trainerr, testerr;
  float RR_bestobjective = -FLT_MAX;
  float RR_bestobjective_temp = -FLT_MAX;
  float ILS_bestobjective = -FLT_MAX;
  float ALL_bestobjective = -FLT_MAX;
  float RR_error=0;
  nfeatures = atoi(argv[6]);
  // thresh: p% random rows select
  thresh = strtod(argv[7], NULL);
  thresh2 = strtod(argv[8], NULL);
  short standardize_data = atoi(argv[9]);
  // nprocs: number of processor?
  nprocs = atoi(argv[10]);
  num_iters = atoi(argv[11]);
  jobs = (int) ((rows+nprocs-1)/nprocs);

  /*allocate space*/
  knn = (int*) malloc(7*sizeof(int));
  mind = (float*) malloc(7*sizeof(float));
  uniq_proj_ind = (int*) malloc((rows+1)*sizeof(int));
  dup = (int*) malloc(rows*sizeof(int));
  rows_j = (int*) malloc(cols*sizeof(int));
  column_numbers = (int*) malloc(cols*sizeof(int));
  column_change = (int*) malloc(cols*sizeof(int));
  row_numbers = (int*) malloc(rows*sizeof(int));
  label = (int*) malloc(rows*sizeof(int));
  origlabel = (int*) malloc(rows*sizeof(int));
  test_label = (int*) malloc(testrows*sizeof(int));
  column = (double*) malloc(rows*sizeof(double));
  projection = (double*) malloc(rows*sizeof(double));
  test_projection = (double*) malloc(testrows*sizeof(double));
  sign = (uint8_t*) malloc(rows*sizeof(uint8_t));

  test_sign = (short*) malloc(testrows*sizeof(short));
  for(i=0;i<testrows;i++) test_sign[i]=0;

  w = (double*) malloc(cols*sizeof(double));
  w_ils = (double*) malloc(cols*sizeof(double));
  w_rr = (double*) malloc(cols*sizeof(double));
  initial_w = (double*) malloc(cols*sizeof(double));

  data = (double**) malloc(rows*sizeof(double*));
  for(i = 0; i < rows; i++)
    data[i] = (double*) malloc(cols*sizeof(double));
  testdata = (double**) malloc(testrows*sizeof(double*));
  for(i = 0; i < testrows; i++)
    testdata[i] = (double*) malloc(cols*sizeof(double));

  newtrain = (int8_t**) malloc(rows*sizeof(int8_t*));
  newtest = (int8_t**) malloc(testrows*sizeof(int8_t*));
  for(i = 0; i < rows; i++)
    newtrain[i] = (int8_t*) malloc(nfeatures*sizeof(int8_t));
  for(i = 0; i < testrows; i++)
    newtest[i] = (int8_t*) malloc(nfeatures*sizeof(int8_t));

  len = (int*) malloc(nfeatures*sizeof(int));
  featuremap = (int*) malloc(cols*sizeof(int));
  allzeros = (short*) malloc(cols*sizeof(short));
  allsame = (short*) malloc(cols*sizeof(short));
  for(i = 0; i < cols; i++){ allzeros[i]=1; allsame[i]=1; }
  rowmap = (int*) malloc(rows*sizeof(int));

  allw0 = (double*) malloc(rows*sizeof(double));

  // Read training data
  FILE* f=fopen(argv[3], "rt");
  for(i=0;i<rows;i++) {
    fscanf(f, "%d", &k);
    origlabel[i] = k;
    if(origlabel[i] == 1) origplus++; else origminus++;
    for(j=0;j<cols;j++)
      fscanf(f, "%lf", &data[i][j]);
  }

  // Read testdata
  f=fopen(argv[5], "rt");
  for(i=0;i<testrows;i++) {
    fscanf(f, "%d", &k);
    test_label[i] = k;
    for(j=0;j<cols;j++)
      fscanf(f, "%lf", &testdata[i][j]);
  }

  if(standardize_data) standardize(data, testdata, rows, cols, testrows);
  getzeroandsame(data, rows, cols);
//  for(i=0; i<cols; i++) { allzeros[i] = 0; allsame[i] = 0; }

  //set default values
  w_inc = 100;
  initialw = 1;
  stopthresh = .001;
  localitthresh = 100;
  localitthresh_ = 10;

// v1 params:
/*  w_inc = 100;
  initialw = 1;
  stopthresh = .01;
  localitthresh = 10;
  localitthresh_ = 10; */

// v2 params:
/*  w_inc = 100;
  initialw = 1;
  stopthresh = 0;
  localitthresh = 100;
  localitthresh_ = 10; */

  // origplus： the number of + class; origminus: the number of - class
  printf("origplus=%f, origminus=%f\n", origplus, origminus);
  printf("w_inc=%f, initialw=%f, stopthresh=%f, localitthresh=%d, localitthresh_=%d, thresh=%f, thresh2=%f, nfeatures=%d, num_iters=%d\n", w_inc, initialw, stopthresh, localitthresh, localitthresh_, thresh, thresh2, nfeatures, num_iters);
  fflush(stdout);

  srand(time(NULL));
  for(i=0; i<nfeatures; i++){

    //permute columns and rows
    cols = origcols;
    for(j = 0; j < cols; j++){ column_change[j] = 1; }
    for(j = 0; j < cols; j++){ column_numbers[j] = j; }
    for(j = 0; j < rows; j++){ row_numbers[j] = j; }
//    fisher_yates(column_numbers, cols-1);
    fisher_yates(row_numbers, rows-1);
    for(j=0;j<cols;j++) featuremap[j] = column_numbers[j];
    for(j=0;j<rows;j++) label[j]=origlabel[j];

//    rd = rand(); rd = rd / (double)RAND_MAX; rd *= cols;
//    if(rd < 1) cols = 1; else cols = (int) rd;
    cols = (int)(thresh2*cols)+1; rd=cols;
    if(rd < 5) cols = 5; else cols = (int) rd;
    if(cols > origcols) cols = origcols;

//select random rows
    k=0;
    size =(int)(thresh*origplus)+1;
//    rd = rand(); rd = rd / (double)RAND_MAX; rd *= size; size = (int) rd;
    if(size < 2) size = 2; if(size > origplus) size = origplus;
    j=0; plus=0;
    while(plus < size){
      if(label[row_numbers[j]] == 1){
        plus++; rowmap[k] = row_numbers[j]; k++;
      }
      j++;
    }
    size =(int)(thresh*origminus)+1;
//    rd = rand(); rd = rd / (double)RAND_MAX; rd *= size; size = (int) rd;
    if(size < 2) size = 2;  if(size > origminus) size = origminus;
    j=0; minus=0;
    while(minus < size){
      if(label[row_numbers[j]] == -1){
        minus++; rowmap[k] = row_numbers[j]; k++;
      }
      j++;
    }
    rows = k;

//coordinate descent
    get_best_w_and_w0();

    rows = origrows;
    for(j=0;j<rows;j++) label[j]=origlabel[j];

    if(bestobjective > RR_bestobjective_temp) {
      RR_bestobjective_temp = bestobjective;
    }

    ILS_bestobjective = bestobjective;
    for(k = 0; k < cols; k++) w_ils[featuremap[k]] = w[featuremap[k]];
    w0_ils = w0;

//evaluate 0/1 loss on all the data
    rows = origrows; plus=origplus; minus=origminus;

    for(j=0;j<rows;j++) label[j]=origlabel[j];
    cols = origcols;

    w_length = 0;
    for(k = 0; k < cols; k++) { w_length += w[k]*w[k]; }
    w_length = sqrt(w_length);

#pragma omp parallel num_threads(nprocs)
    obtain_projection_parallel(jobs, 0);

    sort_projection();
    obtain_unique_projection_values();
    get_full_error(0);
    ALL_bestobjective = bestobjective;

    printf("Initial objective=%f OVERALL objective=%f\n", ILS_bestobjective, ALL_bestobjective);

    for(k = 0; k < cols; k++) w_ils[featuremap[k]] = w[featuremap[k]];
    w0_ils = w0;

//ils permutation - comment if no ils and uncomment otherwise
    ils=1;

    for(l=0; l<num_iters; l++) {

      cols = origcols;

      /* permute rows and columns */
      for(j = 0; j < cols; j++){ column_numbers[j] = j; }
//      fisher_yates(column_numbers, cols-1);
      for(j=0;j<cols;j++) featuremap[j] = column_numbers[j];

//      rd = rand(); rd = rd / (double)RAND_MAX; rd *= cols;
//      if(rd < 1) cols = 1; else cols = (int) rd;
      cols = (int)(thresh2*cols)+1; rd=cols;
      if(rd < 5) cols = 5; else cols = (int) rd;
      if(cols > origcols) cols = origcols;

      for(j = 0; j < rows; j++){ row_numbers[j] = j; }
      fisher_yates(row_numbers, rows-1);
      for(j=0;j<rows;j++) label[j]=origlabel[j];

//randomly select rows
      k=0;
      size =(int)(thresh*origplus)+1;
//      rd = rand(); rd = rd / (double)RAND_MAX; rd *= size; size = (int) rd;
      if(size < 2) size = 2; if(size > origplus) size = origplus;
      j=0; plus=0;
      while(plus < size){
        if(label[row_numbers[j]] == 1){
          plus++; rowmap[k] = row_numbers[j]; k++;
        }
        j++;
      }
      size =(int)(thresh*origminus)+1;
//      rd = rand(); rd = rd / (double)RAND_MAX; rd *= size; size = (int) rd;
      if(size < 2) size = 2;  if(size > origminus) size = origminus;
      j=0; minus=0;
      while(minus < size){
        if(label[row_numbers[j]] == -1){
          minus++; rowmap[k] = row_numbers[j]; k++;
        }
        j++;
      }
      rows = k;

//coordinate descent on selected rows
      get_best_w_and_w0();

      ILS_bestobjective = bestobjective;

//      rows = origrows;
//      for(j=0;j<rows;j++) label[j]=origlabel[j];

//evaluate 0/1 loss on all the data
      rows = origrows; plus=origplus; minus=origminus;

      for(j=0;j<rows;j++) label[j]=origlabel[j];
      cols = origcols;

      w_length = 0;
      for(k = 0; k < cols; k++) { w_length += w[k]*w[k]; }
      w_length = sqrt(w_length);

#pragma omp parallel num_threads(nprocs)
      obtain_projection_parallel(jobs, 0);

      sort_projection();
      obtain_unique_projection_values();
      get_full_error(0);

      if(bestobjective > ALL_bestobjective) {
//      if(1) {
//        ILS_bestobjective = bestobjective;
	ALL_bestobjective = bestobjective;
        for(k = 0; k < cols; k++) w_ils[featuremap[k]] = w[featuremap[k]];
	w0_ils = w0;

      }

      printf("SCD objective=%f ALL objective=%f\n", ILS_bestobjective, ALL_bestobjective);
    }
//    ils=0;

//evaluate 0/1 loss on all the data
    rows = origrows; plus=origplus; minus=origminus;
    for(j=0;j<rows;j++) label[j]=origlabel[j];

    cols = origcols;
    for(k = 0; k < cols; k++) { w[k] = w_ils[k]; }
    w0 = w0_ils;

    w_length = 0;
    for(k = 0; k < cols; k++) { w_length += w[k]*w[k]; }
    w_length = sqrt(w_length);

#pragma omp parallel num_threads(nprocs)
    obtain_projection_parallel(jobs, 0);

    sort_projection();
    obtain_unique_projection_values();
    get_full_error(0);

/*
//optional, disable if not helping
    scan_initial_projection(0, 0);
    get_best_w0(1, uniq_proj_ind_n - 2);
    w0=bestw0;
*/

    ILS_bestobjective = bestobjective;
//    w0_ils=w0;

    for(j = 0; j < testrows; j++) {
      test_proj = 0;
      for(k = 0; k < cols; k++)
        test_proj += testdata[j][k] * w[k];
      if((test_proj/w_length)+w0 <= 0) test_sign[j] += -1; else test_sign[j] += 1;
    }

    printf("Trial %d bestobjective=%f w0=%f\n", i, bestobjective, w0);

    printf("Best w and w0:\n");
    if(w[0] == 0) printf("0"); else printf("%.12f", w[0]/w_length);
    for(k = 1; k < cols; k++) if(w[k] == 0) printf(" 0"); else printf(" %.12f", w[k]/w_length);
    printf("\n");
    printf("%.12f\n", w0);

    if(ILS_bestobjective > RR_bestobjective) {
      for(k = 0; k < cols; k++) w_rr[k] = w_ils[k];
      w0_rr = w0_ils;
      RR_bestobjective = ILS_bestobjective;
      printf("RR bestobjective=%f\n", RR_bestobjective);
      fflush(stdout);
      /* if we reach zero error then exit */
      if(RR_bestobjective == 0) { i = nfeatures; }
    }



//    printf("%d %f %f\n", i+1, -1*RR_bestobjective_temp, -1*RR_bestobjective);
//    for(j = 0; j < cols; j++) { printf("%d (%.5f) ", column_change[j], (1-(float)1/column_change[j])); } printf("\n");

    for(j = 0; j < cols; j++) {
      if(allzeros[j] || allsame[j]) continue;
//      if(rand()/(double)RAND_MAX > .5) {
//      if(rand()/(double)RAND_MAX > (1-(float)1/column_change[j]) ) {
      if(1) {
        double rd = rand(); rd = rd / (double)RAND_MAX; rd *= 2*initialw; rd -= initialw;
        w[j] = rd;
      }
    }

  }

/* in this version all cols are used and so featuremap[k] is really not required */
  w_length = 0;
  for(k = 0; k < cols; k++) { w_length += w_rr[k]*w_rr[k]; }
  w_length = sqrt(w_length);
  printf("Final best w and w0:\n");
  if(w_rr[0] == 0) printf("0"); else printf("%.12f", w_rr[0]/w_length);
  for(k = 1; k < cols; k++) if(w_rr[k] == 0) printf(" 0"); else printf(" %.12f", w_rr[k]/w_length);
  printf("\n");
  printf("%.12f\n", w0_rr);

  for(j = 0; j < testrows; j++) if(test_sign[j] < 0) test_sign[j]=-1; else test_sign[j]=1;

  for(j = 0; j < testrows; j++) {
    test_proj = 0;
    for(k = 0; k < cols; k++) {
      if(allzeros[k] || allsame[k]) assert(w_rr[k]==0);
      test_proj += testdata[j][k] * w_rr[k];
    }
    if((test_proj/w_length)+w0_rr <= 0) newtest[j][0]=-1; else newtest[j][0]=1;
  }

  RR_error = 0;
  for(j = 0; j < testrows; j++)
//    if(test_label[j] != newtest[j][0])
    if(test_label[j] != test_sign[j])
      RR_error++;
    RR_error/=(float)testrows;
  printf("%f\n", RR_error);
  return 0;

}
