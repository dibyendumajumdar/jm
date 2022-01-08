package org.redukti.jm.minpack;

/* http://www.netlib.org/minpack/ex/file15 */


public class Testhybrd2 implements MinPack.Fcnder {

// epsmch is the machine precision

    static final double epsmch = 2.22044604926e-16;
    static final double zero =    0.0;
    static final double half =     .5;
    static final double one =     1.0;
    static final double two =     2.0;
    static final double three =   3.0;
    static final double four =    4.0;
    static final double five =    5.0;
    static final double seven =   7.0;
    static final double eight =   8.0;
    static final double ten =    10.0;
    static final double twenty = 20.0;
    static final double twntf =  25.0;

    int nfev = 0;
    int njev = 0;
    int nprob =0;

/*

Here is a portion of the documentation of the FORTRAN version
of this test code:

c     **********
c
c     this program tests codes for the least-squares solution of
c     m nonlinear equations in n variables. it consists of a driver
c     and an interface subroutine fcn. the driver reads in data,
c     calls the nonlinear least-squares solver, and finally prints
c     out information on the performance of the solver. this is
c     only a sample driver, many other drivers are possible. the
c     interface subroutine fcn is necessary to take into account the
c     forms of calling sequences used by the function and jacobian
c     subroutines in the various nonlinear least-squares solvers.
c
c     subprograms called
c
c       user-supplied ...... fcn
c
c       minpack-supplied ... dpmpar,enorm,initpt,lmder1,ssqfcn
c
c       fortran-supplied ... Math.sqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********

*/


    public static void main (String args[]) {

        int iii,i,ic,k,m,n,nread,ntries,nwrite;

        int info;

//      int nprob;
//      int nfev[] = new int[2];
//      int njev[] = new int[2];

        int[] nprobfile = new int[29];
        int[] nfile = new int[29];
        int[] mfile = new int[29];
        int[] ntryfile = new int[29];

        int[] ma = new int[61];
        int[] na = new int[61];
        int[] nf = new int[61];
        int[] nj = new int[61];
        int[] np = new int[61];
        int[] nx = new int[61];

        double factor,fnorm1,fnorm2,tol;

        double[] fjac = new double[66*41];

        double[] fnm = new double[61];
        double[] fvec = new double[66];
        double[] x = new double[41];

        int[] ipvt = new int[41];

        int num5, ilow, numleft;


//  The npa, na, ma, and nta values are the
//  nprob, n, m, and ntries values from the testdata file.

        nprobfile[1] = 1;
        nprobfile[2] = 1;
        nprobfile[3] = 2;
        nprobfile[4] = 2;
        nprobfile[5] = 3;
        nprobfile[6] = 3;
        nprobfile[7] = 4;
        nprobfile[8] = 5;
        nprobfile[9] = 6;
        nprobfile[10] = 7;
        nprobfile[11] = 8;
        nprobfile[12] = 9;
        nprobfile[13] = 10;
        nprobfile[14] = 11;
        nprobfile[15] = 11;
        nprobfile[16] = 11;
        nprobfile[17] = 12;
        nprobfile[18] = 13;
        nprobfile[19] = 14;
        nprobfile[20] = 15;
        nprobfile[21] = 15;
        nprobfile[22] = 15;
        nprobfile[23] = 15;
        nprobfile[24] = 16;
        nprobfile[25] = 16;
        nprobfile[26] = 16;
        nprobfile[27] = 17;
        nprobfile[28] = 18;


        nfile[1] = 5;
        nfile[2] = 5;
        nfile[3] = 5;
        nfile[4] = 5;
        nfile[5] = 5;
        nfile[6] = 5;
        nfile[7] = 2;
        nfile[8] = 3;
        nfile[9] = 4;
        nfile[10] = 2;
        nfile[11] = 3;
        nfile[12] = 4;
        nfile[13] = 3;
        nfile[14] = 6;
        nfile[15] = 9;
        nfile[16] = 12;
        nfile[17] = 3;
        nfile[18] = 2;
        nfile[19] = 4;
        nfile[20] = 1;
        nfile[21] = 8;
        nfile[22] = 9;
        nfile[23] = 10;
        nfile[24] = 10;
        nfile[25] = 30;
        nfile[26] = 40;
        nfile[27] =  5;
        nfile[28] = 11;


        mfile[1] = 10;
        mfile[2] = 50;
        mfile[3] = 10;
        mfile[4] = 50;
        mfile[5] = 10;
        mfile[6] = 50;
        mfile[7] = 2;
        mfile[8] = 3;
        mfile[9] = 4;
        mfile[10] = 2;
        mfile[11] = 15;
        mfile[12] = 11;
        mfile[13] = 16;
        mfile[14] = 31;
        mfile[15] = 31;
        mfile[16] = 31;
        mfile[17] = 10;
        mfile[18] = 10;
        mfile[19] = 20;
        mfile[20] = 8;
        mfile[21] = 8;
        mfile[22] = 9;
        mfile[23] = 10;
        mfile[24] = 10;
        mfile[25] = 30;
        mfile[26] = 40;
        mfile[27] = 33;
        mfile[28] = 65;


        ntryfile[1] = 1;
        ntryfile[2] = 1;
        ntryfile[3] = 1;
        ntryfile[4] = 1;
        ntryfile[5] = 1;
        ntryfile[6] = 1;
        ntryfile[7] = 3;
        ntryfile[8] = 3;
        ntryfile[9] = 3;
        ntryfile[10] = 3;
        ntryfile[11] = 3;
        ntryfile[12] = 3;
        ntryfile[13] = 2;
        ntryfile[14] = 3;
        ntryfile[15] = 3;
        ntryfile[16] = 3;
        ntryfile[17] = 1;
        ntryfile[18] = 1;
        ntryfile[19] = 3;
        ntryfile[20] = 3;
        ntryfile[21] = 1;
        ntryfile[22] = 1;
        ntryfile[23] = 1;
        ntryfile[24] = 3;
        ntryfile[25] = 1;
        ntryfile[26] = 1;
        ntryfile[27] = 1;
        ntryfile[28] = 1;


        tol = Math.sqrt(epsmch);

        ic = 0;

        for (iii = 1; iii <= 28; iii++) {

//         nprob = nprobfile[iii];
            n = nfile[iii];
            m = mfile[iii];
            ntries = ntryfile[iii];

            Testhybrd2 lmdertest = new Testhybrd2();

            lmdertest.nprob = nprobfile[iii];

            factor = one;

            for (k = 1; k <= ntries; k++) {

                ic++;

                Testhybrd2.initpt(n,x,lmdertest.nprob,factor);

                Testhybrd2.ssqfcn(m,n,x,fvec,lmdertest.nprob);

                fnorm1 = MinPack.enorm(m,0,fvec);

                System.out.print("\n\n\n\n\n problem " + lmdertest.nprob +
                        ", dimensions:  " + n + "  " + m + "\n");
//            write (nwrite,60) nprob,n,m
//   60 format ( //// 5x, 8h problem, i5, 5x, 11h dimensions, 2i5, 5x //
//     *         )

                lmdertest.nfev = 0;
                lmdertest.njev = 0;

                int lwa = n * 5 + m;
                double[] wa = new double[lwa];
                info = MinPack.lmder1(lmdertest,null,m,n,x,fvec,fjac,m,tol,ipvt,wa,lwa);

                ssqfcn(m,n,x,fvec,lmdertest.nprob);

                fnorm2 = MinPack.enorm(m,0,fvec);

                np[ic] = lmdertest.nprob;
                na[ic] = n;
                ma[ic] = m;
                nf[ic] = lmdertest.nfev;
                nj[ic] = lmdertest.njev;
                nx[ic] = info;

                fnm[ic] = fnorm2;

                System.out.print("\n Initial L2 norm of the residuals: " + fnorm1 +
                        "\n Final L2 norm of the residuals: " + fnorm2 +
                        "\n Number of function evaluations: " + lmdertest.nfev +
                        "\n Number of Jacobian evaluations: " + lmdertest.njev +
                        "\n Info value: " + info +
                        "\n Final approximate solution: \n\n");

                num5 = n/5;

                for (i = 1; i <= num5; i++) {

                    ilow = (i-1)*5;
                    System.out.print(x[ilow+1-1] + "  " + x[ilow+2-1] + "  " +
                            x[ilow+3-1] + "  " + x[ilow+4-1] + "  " +
                            x[ilow+5-1] + "\n");

                }

                numleft = n%5;
                ilow = n - numleft;

                switch (numleft) {

                    case 1:

                        System.out.print(x[ilow+1] + "\n\n");

                        break;

                    case 2:

                        System.out.print(x[ilow+1] + "  " + x[ilow+2] + "\n\n");

                        break;

                    case 3:

                        System.out.print(x[ilow+1] + "  " + x[ilow+2] + "  " +
                                x[ilow+3] + "\n\n");

                        break;

                    case 4:

                        System.out.print(x[ilow+1] + "  " + x[ilow+2] + "  " +
                                x[ilow+3] + "  " + x[ilow+4] + "\n\n");

                        break;

                }

//            write (nwrite,70)
//     *            fnorm1,fnorm2,nfev,njev,info,(x(i), i = 1, n)
//   70 format (5x, 33h initial l2 norm of the residuals, d15.7 // 5x,
//     *        33h final l2 norm of the residuals  , d15.7 // 5x,
//     *        33h number of function evaluations  , i10 // 5x,
//     *        33h number of jacobian evaluations  , i10 // 5x,
//     *        15h exit parameter, 18x, i10 // 5x,
//     *        27h final approximate solution // (5x, 5d15.7))

                factor *= ten;

            }

        }

        System.out.print("\n\n\n Summary of " + ic +
                " calls to lmder1: \n\n");
//      write (nwrite,80) ic
//   80 format (12h1summary of , i3, 16h calls to lmder1 /)

        System.out.print("\n\n nprob   n    m   nfev  njev  info  final L2 norm \n\n");

//      write (nwrite,90)
//   90 format (49h nprob   n    m   nfev  njev  info  final l2 norm /)

        for (i = 1; i <= ic; i++) {

            System.out.print(np[i] + "  " + na[i] + "  " + ma[i] + "  " +
                    nf[i] + "  " + nj[i] + "  " + nx[i] + "  " + fnm[i] +
                    "\n");

//         write (nwrite,100) np[i],na[i],ma[i],nf[i],nj[i],nx[i],fnm[i]
//  100 format (3i5, 3i6, 1x, d15.7)

        }

    }

    static final class I {
        int m;
        public I(int m) {
            this.m = m;
        }
        public int _(int row, int col) {
            row = row-1;
            col = col-1;
            return col * m + row;
        }
    }

    @Override
    public int apply(Object p, int m, int n, double[] x, double[] fvec, double[] fjac, int ldfjac, int iflag) {


/*

Documentation from the FORTRAN version:

      subroutine fcn(m,n,x,fvec,fjac,ldfjac,iflag)
      integer m,n,ldfjac,iflag
      double precision x(n),fvec(m),fjac(ldfjac,n)
c     **********
c
c     the calling sequence of fcn should be identical to the
c     calling sequence of the function subroutine in the nonlinear
c     least-squares solver. fcn should only call the testing
c     function and jacobian subroutines ssqfcn and ssqjac with
c     the appropriate value of problem number (nprob).
c
c     subprograms called
c
c       minpack-supplied ... ssqfcn,ssqjac
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********

*/

        if (iflag == 1) ssqfcn(m,n,x,fvec,this.nprob);
        if (iflag == 2) ssqjac(m,n,x,fjac,this.nprob);
        if (iflag == 1) this.nfev++;
        if (iflag == 2) this.njev++;

        return 0;

    }


    public static void vecfcn(int n, double x[], double fvec[],
                              int nprob) {

/*

C     SUBROUTINE VECFCN
C
C     THIS SUBROUTINE DEFINES FOURTEEN TEST FUNCTIONS. THE FIRST
C     FIVE TEST FUNCTIONS ARE OF DIMENSIONS 2,4,2,4,3, RESPECTIVELY,
C     WHILE THE REMAINING TEST FUNCTIONS ARE OF VARIABLE DIMENSION
C     N FOR ANY N GREATER THAN OR EQUAL TO 1 (PROBLEM 6 IS AN
C     EXCEPTION TO THIS, SINCE IT DOES NOT ALLOW N = 1).
C
C     THE SUBROUTINE STATEMENT IS
C
C       SUBROUTINE VECFCN(N,X,FVEC,NPROB)
C
C     WHERE
C
C       N IS A POSITIVE INTEGER INPUT VARIABLE.
C
C       X IS AN INPUT ARRAY OF LENGTH N.
C
C       FVEC IS AN OUTPUT ARRAY OF LENGTH N WHICH CONTAINS THE NPROB
C         FUNCTION VECTOR EVALUATED AT X.
C
C       NPROB IS A POSITIVE INTEGER INPUT VARIABLE WHICH DEFINES THE
C         NUMBER OF THE PROBLEM. NPROB MUST NOT EXCEED 14.
C
C     SUBPROGRAMS CALLED
C
C       FORTRAN-SUPPLIED ... DATAN,DCOS,DEXP,DSIGN,DSIN,DSQRT,
C                            MAX0,MIN0
C
C     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. MARCH 1980.
C     BURTON S. GARBOW, KENNETH E. HILLSTROM, JORGE J. MORE

*/

        int i,iev,ivar,j,k,k1,k2,kp1,ml,mu;
        double c1=1e4,c2=1.0001,c3=2.0e2,c4=2.02e1,c5=1.98e1,c6=1.8e2,c7=2.5e-1,c8=5.0e-1,c9=2.9e1;

        double v[] = new double[12];

// Jacobian routine selector.

        switch (nprob) {

            case 1:

// ROSENBROCK FUNCTION.

                fvec[0] = one - x[0];
                fvec[1] = ten*(x[1] - x[0]*x[0]);

                return;

            case 2:

// POWELL SINGULAR FUNCTION.

                fvec[0] = x[0] + ten*x[1];
                fvec[1] = Math.sqrt(five)*(x[2] - x[3]);
                fvec[2] = (x[1] - two*x[2])*(x[1] - two*x[2]);
                fvec[3] = Math.sqrt(ten)*(x[0] - x[3])*(x[0] - x[3]);

                return;

            case 3:

// POWELL BADLY SCALED FUNCTION.

                fvec[0] = c1*x[0]*x[1] - one;
                fvec[1] = Math.exp(-x[0]) + Math.exp(-x[1]) - c2;

                return;

            case 4: {


                // WOOD FUNCTION.

                double temp1 = x[1] - x[0] * x[0];
                double temp2 = x[3] - x[2] * x[2];
                fvec[0] = -c3 * x[0] * temp1 - (one - x[0]);
                fvec[1] = c3 * temp1 + c4 * (x[1] - one) + c5 * (x[3] - one);
                fvec[2] = -c6 * x[2] * temp2 - (one - x[2]);
                fvec[3] = c6 * temp2 + c4 * (x[3] - one) + c5 * (x[1] - one);

                return;
            }

            case 5: {

// HELICAL VALLEY FUNCTION.

                double tpi = eight * Math.atan(one);
                double temp1;
                if (x[1] < 0.0) {
                    temp1 = -c7;
                } else {
                    temp1 = c7;
                }
                if (x[0] > zero) temp1 = Math.atan(x[1] / x[0]) / tpi;
                if (x[0] < zero) temp1 = Math.atan(x[1] / x[0]) / tpi + c8;
                double temp2 = Math.sqrt(x[0] * x[0] + x[1] * x[1]);
                fvec[0] = ten * (x[2] - ten * temp1);
                fvec[1] = ten * (temp2 - one);
                fvec[2] = x[2];

                return;

            }

            case 6: {

// WATSON FUNCTION.

                for (k = 1; k <= n; k++) {
                    fvec[k-1] = zero;
                }

                for (i = 1; i <= 29; i++) {

                    double ti = i/c9;
                    double sum1 = zero;
                    double temp = one;

                    for (j = 2; j <= n; j++) {
                        sum1 += (double)(j-1)*temp*x[j-1];
                        temp *= ti;
                    }

                    double sum2 = zero;
                    temp = one;

                    for (j = 1; j <= n; j++) {
                        sum2 += temp*x[j-1];
                        temp *= ti;
                    }

                    double temp1 = sum1 - sum2*sum2 - one;
                    double temp2 = two*ti*sum2;
                    temp = one/ti;

                    for (k = 1; k <=n; k++) {
                        fvec[k-1] = fvec[k-1] + temp*((double)(k-1) - temp2)*temp1;
                        temp = ti*temp;
                    }
                }

                double temp = x[1] - x[0]*x[0] - one;
                fvec[0] = fvec[0] + x[0]*(one - two*temp);
                fvec[1] = fvec[1] + temp;

                return;

            }

            case 7: {

//  CHEBYQUAD FUNCTION.

                for (k = 1; k <= n; k++) {
                    fvec[k-1] = zero;
                }
                for (j = 1; j <= n; j++) {
                    double tmp1 = one;
                    double tmp2 = two*x[j-1] - one;
                    double temp = two*tmp2;

                    for (i = 1; i <= n; i++) {
                        fvec[i-1] += tmp2;
                        double ti = temp*tmp2 - tmp1;
                        tmp1 = tmp2;
                        tmp2 = ti;
                    }
                }

                double dx = one/(double)n;
                iev = -1;

                for (k = 1; k <= n; k++) {
                    fvec[k-1] *= dx;
                    if (iev > 0) fvec[k-1] += one/((double)(k*k) - one);
                    iev = -iev;
                }

                return;

            }

            case 8: {

// BROWN ALMOST-LINEAR FUNCTION.

                double sum = -((double)n+1.0);
                double prod = one;

                for (j = 1; j <= n; j++) {
                    sum += x[j-1];
                    prod *= x[j-1];
                }

                for (k = 1; k <= n; k++) {
                    fvec[k-1] = x[k-1] + sum;
                }

                fvec[n-1] = prod - one;

                return;

            }

            case 9: {

// DISCRETE BOUNDARY VALUE FUNCTION.

                double h = one/(double)(n+1);
                for (k =1; k <= n; k++) {
                    double t = x[k-1] + (double)(k) * h + one;
                    double temp = t*t*t;
                    double temp1 = zero;
                    if (k != 1)
                        temp1 = x[k-2];
                    double temp2 = zero;
                    if (k != n)
                        temp2 = x[k];
                    fvec[k-1] = two * x[k-1] - temp1 - temp2 + temp * h*h / two;
                }

                return;

            }

        }

    }


    /**
     * C     SUBROUTINE INITPT
     * C
     * C     THIS SUBROUTINE SPECIFIES THE STANDARD STARTING POINTS FOR
     * C     THE FUNCTIONS DEFINED BY SUBROUTINE VECFCN. THE SUBROUTINE
     * C     RETURNS IN X A MULTIPLE (FACTOR) OF THE STANDARD STARTING
     * C     POINT. FOR THE SIXTH FUNCTION THE STANDARD STARTING POINT IS
     * C     ZERO, SO IN THIS CASE, IF FACTOR IS NOT UNITY, THEN THE
     * C     SUBROUTINE RETURNS THE VECTOR  X(J) = FACTOR, J=1,...,N.
     * C
     * C     THE SUBROUTINE STATEMENT IS
     * C
     * C       SUBROUTINE INITPT(N,X,NPROB,FACTOR)
     * C
     * C     WHERE
     * C
     * C       N IS A POSITIVE INTEGER INPUT VARIABLE.
     * C
     * C       X IS AN OUTPUT ARRAY OF LENGTH N WHICH CONTAINS THE STANDARD
     * C         STARTING POINT FOR PROBLEM NPROB MULTIPLIED BY FACTOR.
     * C
     * C       NPROB IS A POSITIVE INTEGER INPUT VARIABLE WHICH DEFINES THE
     * C         NUMBER OF THE PROBLEM. NPROB MUST NOT EXCEED 14.
     * C
     * C       FACTOR IS AN INPUT VARIABLE WHICH SPECIFIES THE MULTIPLE OF
     * C         THE STANDARD STARTING POINT. IF FACTOR IS UNITY, NO
     * C         MULTIPLICATION IS PERFORMED.
     * C
     * C     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. MARCH 1980.
     * C     BURTON S. GARBOW, KENNETH E. HILLSTROM, JORGE J. MORE
     */
    public static void initpt(int n, double x[], int nprob, double factor) {

        int j;

        double c1 = 1.2,h;


// Selection of initial point.


        switch (nprob) {

            case 1:

// ROSENBROCK FUNCTION.

                x[0] = -c1;
                x[1] = one;

                break;

            case 2:

// POWELL SINGULAR FUNCTION.

                x[0] = three;
                x[1] = -one;
                x[2] = zero;
                x[3] = one;

                break;

// POWELL BADLY SCALED FUNCTION.
            case 3:

                x[0] = zero;
                x[1] = one;

                break;

// WOOD FUNCTION.
            case 4:

                x[0] = -three;
                x[1] = -one;
                x[2] = -three;
                x[3] = -one;

                break;

            case 5:

// HELICAL VALLEY FUNCTION.

                x[0] = -one;
                x[1] = zero;
                x[2] = zero;

                break;

            case 6:

// WATSON FUNCTION.

                for (j = 1; j <= n; j++) {
                    x[j-1] = zero;
                }

                break;

            case 7:

// CHEBYQUAD FUNCTION.

                h = one/(double)(n+1);
                for (j = 1; j <= n; j++) {
                    x[j-1] = j*h;
                }

                break;

            case 8:

// BROWN ALMOST-LINEAR FUNCTION.

                for (j = 1; j <= n; j++) {
                    x[j-1] = half;
                }

                break;

            case 9:

// DISCRETE BOUNDARY VALUE AND INTEGRAL EQUATION FUNCTIONS.

                h = one/(double)(n+1);
                for (j = 1; j <= n; j++) {
                    double tj = (double) j * h;
                    x[j-1] = tj * (tj - one);
                }

                break;

            case 10:

// TRIGONOMETRIC FUNCTION.

                h = one/(double)(n);
                for (j = 1; j <= n; j++) {
                    x[j-1] = h;
                }

                break;

            case 11:

// VARIABLY DIMENSIONED FUNCTION.

                h = one/(double)(n);
                for (j = 1; j <= n; j++) {
                    x[j-1] = one - (double)(j) * h;
                }

                break;

            case 12:

// BROYDEN TRIDIAGONAL AND BANDED FUNCTIONS.

                for (j = 1; j <= n; j++) {
                    x[j-1] = -one;
                }

                break;

        }

// Compute multiple of initial point.

        if (factor == one) return;

        if (nprob != 6) {

            for (j = 1; j <= n; j++) {
                x[j-1] *= factor;
            }

        } else {

            for (j = 1; j <= n; j++) {
                x[j-1] = factor;
            }

        }

        return;

    }



    public static void ssqfcn(int m, int n, double x[],
                              double fvec[], int nprob) {


/*

Documentaton from the FORTRAN version:


      subroutine ssqfcn(m,n,x,fvec,nprob)
      integer m,n,nprob
      double precision x(n),fvec(m)
c     **********
c
c     subroutine ssqfcn
c
c     this subroutine defines the functions of eighteen nonlinear
c     least squares problems. the allowable values of (m,n) for
c     functions 1,2 and 3 are variable but with m .ge. n.
c     for functions 4,5,6,7,8,9 and 10 the values of (m,n) are
c     (2,2),(3,3),(4,4),(2,2),(15,3),(11,4) and (16,3), respectively.
c     function 11 (watson) has m = 31 with n usually 6 or 9.
c     however, any n, n = 2,...,31, is permitted.
c     functions 12,13 and 14 have n = 3,2 and 4, respectively, but
c     allow any m .ge. n, with the usual choices being 10,10 and 20.
c     function 15 (chebyquad) allows m and n variable with m .ge. n.
c     function 16 (brown) allows n variable with m = n.
c     for functions 17 and 18, the values of (m,n) are
c     (33,5) and (65,11), respectively.
c
c     the subroutine statement is
c
c       subroutine ssqfcn(m,n,x,fvec,nprob)
c
c     where
c
c       m and n are positive integer input variables. n must not
c         exceed m.
c
c       x is an input array of length n.
c
c       fvec is an output array of length m which contains the nprob
c         function evaluated at x.
c
c       nprob is a positive integer input variable which defines the
c         number of the problem. nprob must not exceed 18.
c
c     subprograms called
c
c       fortran-supplied ... Math.atan,Math.cos,dexp,Math.sin,Math.sqrt,dsign
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********

*/


        int i,iev,j,nm1;

        double c13,c14,c29,c45,div,dx,prod,sum,
                s1,s2,temp,ti,tmp1,tmp2,tmp3,tmp4,tpi,
                zp5;

        zp5 = .5;
        c13 = 13.0;
        c14 = 14.0;
        c29 = 29.0;
        c45 = 45.0;

        double v[] =
                {4.0e0,2.0e0,1.0e0,5.0e-1,2.5e-1,1.67e-1,1.25e-1,1.0e-1,
                        8.33e-2,7.14e-2,6.25e-2};

        double y1[] =
                {1.4e-1,1.8e-1,2.2e-1,2.5e-1,2.9e-1,3.2e-1,3.5e-1,3.9e-1,
                        3.7e-1,5.8e-1,7.3e-1,9.6e-1,1.34e0,2.1e0,4.39e0};

        double y2[] =
                {1.957e-1,1.947e-1,1.735e-1,1.6e-1,8.44e-2,6.27e-2,4.56e-2,
                        3.42e-2,3.23e-2,2.35e-2,2.46e-2};

        double y3[] =
                {3.478e4,2.861e4,2.365e4,1.963e4,1.637e4,1.372e4,1.154e4,
                        9.744e3,8.261e3,7.03e3,6.005e3,5.147e3,4.427e3,3.82e3,
                        3.307e3,2.872e3};

        double y4[] =
                {8.44e-1,9.08e-1,9.32e-1,9.36e-1,9.25e-1,9.08e-1,8.81e-1,
                        8.5e-1,8.18e-1,7.84e-1,7.51e-1,7.18e-1,6.85e-1,6.58e-1,
                        6.28e-1,6.03e-1,5.8e-1,5.58e-1,5.38e-1,5.22e-1,5.06e-1,
                        4.9e-1,4.78e-1,4.67e-1,4.57e-1,4.48e-1,4.38e-1,4.31e-1,
                        4.24e-1,4.2e-1,4.14e-1,4.11e-1,4.06e-1};

        double y5[] =
                {1.366e0,1.191e0,1.112e0,1.013e0,9.91e-1,8.85e-1,8.31e-1,
                        8.47e-1,7.86e-1,7.25e-1,7.46e-1,6.79e-1,6.08e-1,6.55e-1,
                        6.16e-1,6.06e-1,6.02e-1,6.26e-1,6.51e-1,7.24e-1,6.49e-1,
                        6.49e-1,6.94e-1,6.44e-1,6.24e-1,6.61e-1,6.12e-1,5.58e-1,
                        5.33e-1,4.95e-1,5.0e-1,4.23e-1,3.95e-1,3.75e-1,3.72e-1,
                        3.91e-1,3.96e-1,4.05e-1,4.28e-1,4.29e-1,5.23e-1,5.62e-1,
                        6.07e-1,6.53e-1,6.72e-1,7.08e-1,6.33e-1,6.68e-1,6.45e-1,
                        6.32e-1,5.91e-1,5.59e-1,5.97e-1,6.25e-1,7.39e-1,7.1e-1,
                        7.29e-1,7.2e-1,6.36e-1,5.81e-1,4.28e-1,2.92e-1,1.62e-1,
                        9.8e-2,5.4e-2};


// Function routine selector.

        switch (nprob) {

            case 1:

// Linear function - full rank.

                sum = zero;

                for (j = 1; j <= n; j++) {

                    sum += x[j-1];

                }

                temp = two*sum/m + one;

                for (i = 1; i <= m; i++) {

                    fvec[i-1] = -temp;
                    if (i <= n) fvec[i-1] += x[i-1];

                }

                return;

            case 2:

// Linear function - rank 1.

                sum = zero;

                for (j = 1; j <= n; j++) {

                    sum += j*x[j-1];

                }

                for (i = 1; i <= m; i++) {

                    fvec[i-1] = i*sum - one;

                }

                return;

            case 3:

// Linear function - rank 1 with zero columns and rows.

                sum = zero;
                nm1 = n - 1;

                for (j = 2; j <= nm1; j++) {

                    sum += j*x[j-1];

                }

                for (i = 1; i <= m; i++) {

                    fvec[i-1] = (i-1)*sum - one;

                }

                fvec[m-1] = -one;

                return;

            case 4:

// Rosenbrock function.

                fvec[1-1] = ten*(x[2-1] - x[1-1]*x[1-1]);
                fvec[2-1] = one - x[1-1];

                return;

            case 5:

// Helical valley function.

                tpi = eight*Math.atan(one);

                if (x[2-1] < 0.0) {

                    tmp1 = -.25;

                } else {

                    tmp1 = .25;

                }

                if (x[1-1] > zero) tmp1 = Math.atan(x[2-1]/x[1-1])/tpi;
                if (x[1-1] < zero) tmp1 = Math.atan(x[2-1]/x[1-1])/tpi + zp5;
                tmp2 = Math.sqrt(x[1-1]*x[1-1] + x[2-1]*x[2-1]);
                fvec[1-1] = ten*(x[3-1] - ten*tmp1);
                fvec[2-1] = ten*(tmp2 - one);
                fvec[3-1] = x[3-1];

                return;

            case 6:

// Powell singular function.

                fvec[1-1] = x[1-1] + ten*x[2-1];
                fvec[2-1] = Math.sqrt(five)*(x[3-1] - x[4-1]);
//            fvec[3] = (x[2] - two*x[3])*(x[2] - two*x[3]);
//            fvec[4] = Math.sqrt(ten)*(x[1] - x[4])*(x[1] - x[4]);
                fvec[3-1] = Math.pow(x[2-1] - two*x[3-1],2);
                fvec[4-1] = Math.sqrt(ten)*Math.pow(x[1-1] - x[4-1],2);

                return;

            case 7:

// Freudenstein and Roth function.

                fvec[1-1] = -c13 + x[1-1] + ((five - x[2-1])*x[2-1] - two)*x[2-1];
                fvec[2-1] = -c29 + x[1-1] + ((one + x[2-1])*x[2-1] - c14)*x[2-1];

                return;

            case 8:

// Bard function.

                for (i = 1; i <= 15; i++) {

                    tmp1 = i;
                    tmp2 = 16 - i;
                    tmp3 = tmp1;

                    if (i > 8) tmp3 = tmp2;

                    fvec[i-1] = y1[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));

                }

                return;

            case 9:

// Kowalik and Osborne function.

                for (i = 1; i <= 11; i++) {

                    tmp1 = v[i-1]*(v[i-1] + x[2-1]);
                    tmp2 = v[i-1]*(v[i-1] + x[3-1]) + x[4-1];
                    fvec[i-1] = y2[i-1] - x[1-1]*tmp1/tmp2;

                }

                return;

            case 10:

// Meyer function.

                for (i = 1; i <= 16; i++) {

                    temp = five*i + c45 + x[3-1];
                    tmp1 = x[2-1]/temp;
                    tmp2 = Math.exp(tmp1);
                    fvec[i-1] = x[1-1]*tmp2 - y3[i-1];

                }

                return;

            case 11:

// Watson function.

                for (i = 1; i <= 29; i++) {

                    div = i/c29;
                    s1 = zero;
                    dx = one;

                    for (j = 2; j <= n; j++) {

                        s1 += (j-1)*dx*x[j-1];
                        dx *= div;

                    }

                    s2 = zero;
                    dx = one;

                    for (j = 1; j <= n; j++) {

                        s2 += dx*x[j-1];
                        dx *= div;

                    }

                    fvec[i-1] = s1 - s2*s2 - one;

                }

                fvec[30-1] = x[1-1];
                fvec[31-1] = x[2-1] - x[1-1]*x[1-1] - one;

                return;

            case 12:

// Box 3-dimensional function.

                for (i = 1; i <= m; i++) {

                    temp = i;
                    tmp1 = temp/ten;
                    fvec[i-1] = Math.exp(-tmp1*x[1-1]) - Math.exp(-tmp1*x[2-1])
                            + (Math.exp(-temp) - Math.exp(-tmp1))*x[3-1];

                }

                return;

            case 13:

// Jennrich and Sampson function.

                for (i = 1; i <= m; i++) {

                    temp = i;
                    fvec[i-1] = two + two*temp - Math.exp(temp*x[1-1]) - Math.exp(temp*x[2-1]);

                }

                return;

            case 14:

// Brown and Dennis function.

                for (i = 1; i <= m; i++) {

                    temp = i/five;
                    tmp1 = x[1-1] + temp*x[2-1] - Math.exp(temp);
                    tmp2 = x[3-1] + Math.sin(temp)*x[4-1] - Math.cos(temp);
                    fvec[i-1] = tmp1*tmp1 + tmp2*tmp2;

                }

                return;

            case 15:

// Chebyquad function.

                for (i = 1; i <= m; i++) {

                    fvec[i-1] = zero;

                }

                for (j = 1; j <= n; j++) {

                    tmp1 = one;
                    tmp2 = two*x[j-1] - one;
                    temp = two*tmp2;

                    for (i = 1; i <= m; i++) {

                        fvec[i-1] += tmp2;
                        ti = temp*tmp2 - tmp1;
                        tmp1 = tmp2;
                        tmp2 = ti;

                    }

                }

                dx = one/n;
                iev = -1;

                for (i = 1; i <= m; i++) {

                    fvec[i-1] *= dx;
                    if (iev > 0) fvec[i-1] += one/(i*i - one);
                    iev = -iev;

                }

                return;

            case 16:

// Brown almost-linear function.

                sum = -(n+1);
                prod = one;

                for (j = 1; j <= n; j++) {

                    sum += x[j-1];
                    prod *= x[j-1];

                }

                for (i = 1; i <= n; i++) {

                    fvec[i-1] = x[i-1] + sum;

                }

                fvec[n-1] = prod - one;

                return;

            case 17:

// Osborne 1 function.

                for (i = 1; i <= 33; i++) {

                    temp = ten*(i-1);
                    tmp1 = Math.exp(-x[4-1]*temp);
                    tmp2 = Math.exp(-x[5-1]*temp);
                    fvec[i-1] = y4[i-1] - (x[1-1] + x[2-1]*tmp1 + x[3-1]*tmp2);

                }

                return;

            case 18:

// Osborne 2 function.

                for (i = 1; i <= 65; i++) {

                    temp = (i-1)/ten;
                    tmp1 = Math.exp(-x[5-1]*temp);
                    tmp2 = Math.exp(-x[6-1]*(temp-x[9-1])*(temp-x[9-1]));
                    tmp3 = Math.exp(-x[7-1]*(temp-x[10-1])*(temp-x[10-1]));
                    tmp4 = Math.exp(-x[8-1]*(temp-x[11-1])*(temp-x[11-1]));
                    fvec[i-1] = y5[i-1]
                            - (x[1-1]*tmp1 + x[2-1]*tmp2 + x[3-1]*tmp3 + x[4-1]*tmp4);
                }

                return;

        }

    }

}
