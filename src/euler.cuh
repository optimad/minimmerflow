#ifndef __EULERCUH__
#define __EULERCUH__
#include "storage.hpp"
#include <volume_kernel.hpp>
#include "cuda_runtime.h"

namespace CudaWrappers {

    void evalFluxes_wrapper
    (
        const double *conservative,
        const double *primitive, 
        const std::array<double, 3> &n,
        FluxData *fluxes,
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


    void initRHS_wrapper(std::vector<double> *RHSVec);

}
#endif
