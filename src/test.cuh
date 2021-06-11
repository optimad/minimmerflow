#ifndef __TESTCUH__
#define __TESTCUH__
#include "cuda_runtime.h"

namespace CudaWrappers {

    void add_wrapper(int N, double *x, double *y, double *z, int blockSize);

}
#endif
