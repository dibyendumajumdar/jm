package org.redukti.jm.minpack_f77;

public interface Lmdif_fcn {

   void fcn(int m, int n, double x[], double fvec[],
            int iflag[]);

}
