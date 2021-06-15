#ifndef __EULERCUH__
#define __EULERCUH__
#include "storage.hpp"
#include <volume_kernel.hpp>
#include "cuda_runtime.h"

namespace CudaWrappers {

    void initRHS_wrapper(double *RHS, const int N);

    void evalSplitting_wrapper
    (
        std::array<double, N_FIELDS> *ownerReconstruction,
        std::array<double, N_FIELDS> *neighReconstruction,
        std::array<double, 3> *interfaceNormal,
        double *fluxesVec,
        double *maxEig,
        const long Nif,
        const int N_FIELDS,
        const int FID_U,
        const int FID_V,
        const int FID_W,
        const int FID_P,
        const int FID_RHO,
        const int FID_EQ_C,
        const int FID_EQ_M_X,
        const int FID_EQ_M_Y,
        const int FID_EQ_M_Z,
        const int FID_EQ_E,
        const double GAMMA
    );
}
#endif
