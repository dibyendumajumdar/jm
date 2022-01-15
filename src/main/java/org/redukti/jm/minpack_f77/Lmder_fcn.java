package org.redukti.jm.minpack_f77;

public interface Lmder_fcn {

   void fcn(int m, int n, double x[], double fvec[],
            double fjac[][], int iflag[]);

}
