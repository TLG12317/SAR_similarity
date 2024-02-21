/***********************************************************************
* mexeig.c : C mex file 
*
* mex -O -largeArrayDims -lmwlapack -lmwblas mexeig.c
*
* [V,d] = mexeig(A,options); 
* options = 1 (default), compute both eigenvectors and eigenvalues;
*         = 0, compute only eigenvalues.
***********************************************************************/

#include <math.h>
#include <mex.h>
#include <matrix.h>
#include <string.h> /* needed for memcpy() */

#if !defined(MX_API_VER) || ( MX_API_VER < 0x07030000 )
typedef int mwIndex;
typedef int mwSize;
#endif

/**********************************************************
* 
***********************************************************/
void mexFunction(
      int nlhs,   mxArray  *plhs[], 
      int nrhs,   const mxArray  *prhs[] )

{    double   *A, *V, *d, *work, *work2;  

     mwIndex  subs[2];
     mwSize   nsubs=2; 
     mwSize   m, n, lwork, lwork2, info, j, k, jn, options; 
     char     *jobz="V";
     char     *uplo="U"; 

/* CHECK FOR PROPER NUMBER OF ARGUMENTS */

   if (nrhs > 2){
      mexErrMsgTxt("mexeig: requires at most 2 input arguments."); }
   if (nlhs > 2){ 
      mexErrMsgTxt("mexeig: requires at most 2 output argument."); }   

/* CHECK THE DIMENSIONS */

    m = mxGetM(prhs[0]); 
    n = mxGetN(prhs[0]); 
    if (m != n) { 
       mexErrMsgTxt("mexeig: matrix must be square."); }
    if (mxIsSparse(prhs[0])) {
       mexErrMsgTxt("mexeig: sparse matrix not allowed."); }   
    A = mxGetPr(prhs[0]);     
    options = 1; 
    if (nrhs==2) { options = (int)*mxGetPr(prhs[1]); } 
    if (options==1) { jobz="V"; } else { jobz="N"; } 

    /***** create return argument *****/
    plhs[0] = mxCreateDoubleMatrix(n,n,mxREAL); 
    V = mxGetPr(plhs[0]);  
    plhs[1] = mxCreateDoubleMatrix(n,1,mxREAL); 
    d = mxGetPr(plhs[1]);  

    /***** Do the computations in a subroutine *****/
    lwork  = 1+6*n+2*n*n;  
    work   = mxCalloc(lwork,sizeof(double)); 
    lwork2 = 3 + 5*n; 
    work2  = mxCalloc(lwork2,sizeof(double)); 
    /***
    for (j=0;j<n;j++) { 
      jn = j*n; 
      for (k=0;k<=j;k++) { V[k+jn] = A[k+jn]; }
    }
    ***/
    memcpy(mxGetPr(plhs[0]),mxGetPr(prhs[0]),(m*n)*sizeof(double));

    dsyevd(jobz,uplo,&n, V,&n, d, work,&lwork, work2,&lwork2, &info); 

    return;
 }
/**********************************************************/
