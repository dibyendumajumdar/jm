package org.redukti.jm.minpack;

/* http://www.netlib.org/minpack/ex/file15 */

import org.junit.Test;

public class TestHybrd2 implements MinPack.Hybrd_Function {

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
    int nprob =0;

/*

Here is a portion of the documentation of the FORTRAN version
of this test code:

C     THIS PROGRAM TESTS CODES FOR THE SOLUTION OF N NONLINEAR
C     EQUATIONS IN N VARIABLES. IT CONSISTS OF A DRIVER AND AN
C     INTERFACE SUBROUTINE FCN. THE DRIVER READS IN DATA, CALLS THE
C     NONLINEAR EQUATION SOLVER, AND FINALLY PRINTS OUT INFORMATION
C     ON THE PERFORMANCE OF THE SOLVER. THIS IS ONLY A SAMPLE DRIVER,
C     MANY OTHER DRIVERS ARE POSSIBLE. THE INTERFACE SUBROUTINE FCN
C     IS NECESSARY TO TAKE INTO ACCOUNT THE FORMS OF CALLING
C     SEQUENCES USED BY THE FUNCTION SUBROUTINES IN THE VARIOUS
C     NONLINEAR EQUATION SOLVERS.
C
C     SUBPROGRAMS CALLED
C
C       USER-SUPPLIED ...... FCN
C
C       MINPACK-SUPPLIED ... DPMPAR,ENORM,HYBRD1,INITPT,VECFCN
C
C       FORTRAN-SUPPLIED ... DSQRT
C
C     ARGONNE NATIONAL LABORATORY. MINPACK PROJECT. MARCH 1980.
C     BURTON S. GARBOW, KENNETH E. HILLSTROM, JORGE J. MORE

*/

    static final class TestProblem {
        public final int nprob, n, ntries;

        public TestProblem(int nprob, int n, int ntries) {
            this.nprob = nprob;
            this.n = n;
            this.ntries = ntries;
        }
    }

    @Test
    public void hybrdTest()
    {
        TestHybrd2 instance = new TestHybrd2();
        instance.run();
    }


    public void run () {

        TestProblem[] problems = {
                new TestProblem(1,    2,    3),
                new TestProblem(2,    4,    3),
                new TestProblem(3,    2,    2),
                new TestProblem(4,    4,    3),
                new TestProblem(5,    3,    3),
                new TestProblem(6,    6,    2),
                new TestProblem(6,    9,    2),
                new TestProblem(7,    5,    3),
                new TestProblem(7,    6,    3),
                new TestProblem(7,    7,    3),
                new TestProblem(7,    8,    1),
                new TestProblem(7,    9,    1),
                new TestProblem(8,   10,    3),
                new TestProblem(8,   30,    1),
                new TestProblem(8,   40,    1),
                new TestProblem(9,   10,    3),
                new TestProblem(10,    1,    3),
                new TestProblem(10,   10,    3),
                new TestProblem(11,   10,    3),
                new TestProblem(12,   10,    3),
                new TestProblem(13,   10,    3),
                new TestProblem(14,   10,    3)
        };

        int IC,INFO,K,LWA;
        int[] NA = new int[60],
            NF = new int[60],
            NP = new int[60],
            NX = new int[60];
        double FACTOR,FNORM1,FNORM2,TOL;
        double[] FNM = new double[60],
            FVEC = new double[40],
            WA = new double[2660],
            X = new double[40];

        TOL = StrictMath.sqrt(MinPack.dpmpar(1));
        LWA = 2660;
        IC = 0;

        for (TestProblem p: problems) {
            FACTOR = one;
            this.nprob = p.nprob;
            for (K = 1; K <= p.ntries; K++) {
                IC = IC + 1;
                initpt(p.n,X,p.nprob,FACTOR);
                vecfcn(p.n,X,FVEC,p.nprob);
                FNORM1 = MinPack.enorm(p.n, 0, FVEC);
                System.out.println("PROBLEM " + p.nprob + " DIMENSION " + p.n);
                nfev = 0;
                INFO = MinPack.hybrd1(this,p.n,X,FVEC,TOL,WA,LWA);
                FNORM2 = MinPack.enorm(p.n,0,FVEC);
                NP[IC-1] = p.nprob;
                NA[IC-1] = p.n;
                NF[IC-1] = nfev;
                NX[IC-1] = INFO;
                FNM[IC-1] = FNORM2;
                System.out.println("INITIAL L2 NORM OF THE RESIDUALS " + FNORM1);
                System.out.println("FINAL L2 NORM OF THE RESIDUALS " + FNORM2);
                System.out.println("NUMBER OF FUNCTION EVALUATIONS " + nfev);
                System.out.println("EXIT PARAMETER " + INFO);
                System.out.println("FINAL APPROXIMATE SOLUTION");
                for (int i = 0; i < p.n; i++) {
                    System.out.print(X[i] + " ");
                }
                System.out.println();
                System.out.println();
                FACTOR = ten*FACTOR;
            }
        }

        System.out.println("NPROB   N    NFEV  INFO  FINAL L2 NORM");
        for (int i = 0; i < IC; i++) {
            System.out.println(NP[i] + " " + NA[i] + " " + NF[i] + " " + NX[i] + " " + FNM[i]);
        }
/*
      DO 40 I = 1, IC
         WRITE (NWRITE,100) NP(I),NA(I),NF(I),NX(I),FNM(I)
   40    CONTINUE
      STOP
   50 FORMAT (3I5)
   60 FORMAT ( //// 5X, 8H PROBLEM, I5, 5X, 10H DIMENSION, I5, 5X //)
   70 FORMAT (5X, 33H INITIAL L2 NORM OF THE RESIDUALS, D15.7 // 5X,
     *        33H FINAL L2 NORM OF THE RESIDUALS  , D15.7 // 5X,
     *        33H NUMBER OF FUNCTION EVALUATIONS  , I10 // 5X,
     *        15H EXIT PARAMETER, 18X, I10 // 5X,
     *        27H FINAL APPROXIMATE SOLUTION // (5X, 5D15.7))
   80 FORMAT (12H1SUMMARY OF , I3, 16H CALLS TO HYBRD1 /)
   90 FORMAT (39H NPROB   N    NFEV  INFO  FINAL L2 NORM /)
  100 FORMAT (I4, I6, I7, I6, 1X, D15.7)
 */

    }

    @Override
    public void apply(int n, double[] x, double[] fvec, int[] iflag) {
        vecfcn(n,x,fvec,nprob);
        nfev = nfev + 1;
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

        int i,iev,j,k,k1,k2,kp1,ml,mu;
        double c1=1e4,c2=1.0001,c3=2.0e2,c4=2.02e1,c5=1.98e1,c6=1.8e2,c7=2.5e-1,c8=5.0e-1,c9=2.9e1;

        switch (nprob) {

            case 1:

// ROSENBROCK FUNCTION.

                fvec[0] = one - x[0];
                fvec[1] = ten*(x[1] - x[0]*x[0]);

                return;

            case 2:

// POWELL SINGULAR FUNCTION.

                fvec[0] = x[0] + ten*x[1];
                fvec[1] = StrictMath.sqrt(five)*(x[2] - x[3]);
                fvec[2] = (x[1] - two*x[2])*(x[1] - two*x[2]);
                fvec[3] = StrictMath.sqrt(ten)*(x[0] - x[3])*(x[0] - x[3]);

                return;

            case 3:

// POWELL BADLY SCALED FUNCTION.

                fvec[0] = c1*x[0]*x[1] - one;
                fvec[1] = StrictMath.exp(-x[0]) + StrictMath.exp(-x[1]) - c2;

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

                double tpi = eight * StrictMath.atan(one);
                double temp1;
                if (x[1] < 0.0) {
                    temp1 = -c7;
                } else {
                    temp1 = c7;
                }
                if (x[0] > zero) temp1 = StrictMath.atan(x[1] / x[0]) / tpi;
                if (x[0] < zero) temp1 = StrictMath.atan(x[1] / x[0]) / tpi + c8;
                double temp2 = StrictMath.sqrt(x[0] * x[0] + x[1] * x[1]);
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

            case 10: {

// DISCRETE INTEGRAL EQUATION FUNCTION.

                double h = one/(double)(n+1);
                for (k = 1; k <= n; k++) {
                    double tk = (double)(k)*h;
                    double sum1 = zero;
                    for (j = 1; j<= k; j++) {
                        double tj = (double)(j)*h;
                        double t = x[j-1] + tj + one;
                        double temp = t*t*t;
                        sum1 = sum1 + tj*temp;
                    }
                    double sum2 = zero;
                    kp1 = k + 1;
                    if (n >= kp1) {
                        for (j = kp1; j <= n; j++) {
                            double tj = (double)(j)*h;
                            double t = x[j-1] + tj + one;
                            double temp = t*t*t;
                            sum2 = sum2 + (one - tj)*temp;
                        }
                    }
                    fvec[k-1] = x[k-1] + h*((one - tk)*sum1 + tk*sum2)/two;
                }

                return;

            }

            case 11: {

// TRIGONOMETRIC FUNCTION.

                double sum = zero;
                for (j = 1; j <= n; j++) {
                    fvec[j-1] = StrictMath.cos(x[j-1]);
                    sum = sum + fvec[j-1];
                }
                for (k = 1; k <= n; k++) {
                    fvec[k-1] = (double)(n+k) - StrictMath.sin(x[k-1]) - sum - (double)(k)*fvec[k-1];
                }

                return;
            }

            case 12: {

// VARIABLY DIMENSIONED FUNCTION.

                double sum = zero;
                for (j = 1; j <= n; j++) {
                    sum = sum + (double)(j)*(x[j-1] - one);
                }
                double temp = sum*(one + two*sum*sum);
                for (k = 1; k <= n; k++) {
                    fvec[k-1] = x[k-1] - one + (double)(k)*temp;
                }

                return;
            }

            case 13: {

// BROYDEN TRIDIAGONAL FUNCTION.

                for (k = 1; k <= n; k++) {
                    double temp = (three - two*x[k-1])*x[k-1];
                    double temp1 = zero;
                    if (k != 1)
                        temp1 = x[k-2];
                    double temp2 = zero;
                    if (k != n)
                        temp2 = x[k];
                    fvec[k-1] = temp - temp1 - two*temp2 + one;
                }

                return;

            }

            case 14: {

// BROYDEN BANDED FUNCTION.

                ml = 5;
                mu = 1;
                for (k = 1; k<= n; k++) {
                    k1 = Math.max(1,k-ml); // MAX0?
                    k2 = Math.min(k+mu,n);     // MIN0
                    double temp = zero;
                    for (j = k1; j <= k2; j++) {
                        if (j != k)
                            temp = temp + x[j-1]*(one + x[j-1]);
                    }
                    fvec[k-1] = x[k-1]*(two + five*x[k-1]*x[k-1]) + one - temp;
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
            case 10:

// DISCRETE BOUNDARY VALUE AND INTEGRAL EQUATION FUNCTIONS.

                h = one/(double)(n+1);
                for (j = 1; j <= n; j++) {
                    double tj = (double) j * h;
                    x[j-1] = tj * (tj - one);
                }

                break;

            case 11:

// TRIGONOMETRIC FUNCTION.

                h = one/(double)(n);
                for (j = 1; j <= n; j++) {
                    x[j-1] = h;
                }

                break;

            case 12:

// VARIABLY DIMENSIONED FUNCTION.

                h = one/(double)(n);
                for (j = 1; j <= n; j++) {
                    x[j-1] = one - (double)(j) * h;
                }

                break;

            case 13:
            case 14:

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



}
