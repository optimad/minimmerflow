#include "storage.hpp"
#include "volume_kernel.hpp"
#include "patch_kernel.hpp"
#include "voloctree.hpp"

/*!
 * Calculates the conservative fluxes for a perfect gas.
 *
 * \param conservative is the conservative state
 * \param primitive is the primitive state
 * \param n is the normal direction
 * \param[out] fluxes on output will contain the conservative fluxes
 */
__device__
void evalFluxes_cu
(
    const double *conservative,
    const double *primitive, 
          double *fluxes, 
    const double *n,
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
)
{
    // Compute variables
    double u = primitive[FID_U];
    double v = primitive[FID_V];
    double w = primitive[FID_W];

    double vel2 = u * u + v * v + w * w;
    double un = 
        primitive[FID_U] * n[0] 
      + primitive[FID_V] * n[1] 
      + primitive[FID_W] * n[2];

    double p = primitive[FID_P];
//  if (p < 0.) {
//    log::cout() << "***** Negative pressure (" << p << ") in flux computation!\n";
//  }

    double rho = conservative[FID_RHO];
//  if (rho < 0.) {
//      log::cout() << "***** Negative density in flux computation!\n";
//  }

    double eto = p / (GAMMA - 1.) + 0.5 * rho * vel2;

    // Compute fluxes
    double massFlux = rho * un;

    fluxes[FID_EQ_C]   = massFlux;
    fluxes[FID_EQ_M_X] = massFlux * u + p * n[0];
    fluxes[FID_EQ_M_Y] = massFlux * v + p * n[1];
    fluxes[FID_EQ_M_Z] = massFlux * w + p * n[2];
    fluxes[FID_EQ_E]   = un * (eto + p);
}


/*!
 * Initializes the RHS of Euler equations
 *
 * \param pointer to RHS
 * \param size of RHS
 */
__global__
void initializeRHS(double *RHS, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        RHS[i] = 0.;
    }
}


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
    )
    {
        // Allocate on GPU
        double *dprimitive, *dconservative, *dfluxes, *dnvec;
        int size = 5 * sizeof(double);

        cudaMalloc((void **) &dconservative, size);
        cudaMalloc((void **) &dprimitive, size);
        cudaMalloc((void **) &dfluxes, size);
        cudaMemcpy(dconservative, conservative, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dprimitive, primitive, size, cudaMemcpyHostToDevice);


        double *fluxesVec= new double[5];

        int nSize = 3 * sizeof(double);
        double *nvec = new double[3];
        cudaMalloc((void **) &dnvec, nSize);
        nvec[0] = n[0];
        nvec[1] = n[1];
        nvec[2] = n[2];
        cudaMemcpy(dnvec, nvec, nSize, cudaMemcpyHostToDevice);

//      evalFluxes_cu
//      (
//          dconservative, 
//          dprimitive, 
//          dfluxes,
//          dnvec, 
//          FID_U,
//          FID_V,
//          FID_W,
//          FID_P,
//          FID_RHO,
//          FID_EQ_C,
//          FID_EQ_M_X,
//          FID_EQ_M_Y,
//          FID_EQ_M_Z,
//          FID_EQ_E,
//          GAMMA
//      );

        cudaMemcpy(fluxesVec, dfluxes, size, cudaMemcpyDeviceToHost);
        (*fluxes)[0] = fluxesVec[0];
        (*fluxes)[1] = fluxesVec[1];
        (*fluxes)[2] = fluxesVec[2];
        (*fluxes)[3] = fluxesVec[3];
        (*fluxes)[4] = fluxesVec[4];

        cudaFree(dprimitive);
        cudaFree(dconservative);
        cudaFree(dnvec);
        cudaFree(dfluxes);

    }


    void initRHS_wrapper(std::vector<double> *RHSVec)
    {
        int N = RHSVec->size();

        double *RHS = new double[N];
        for(int i = 0; i < RHSVec->size(); i++) {
            RHS[i] = RHSVec->data()[i];
        }

        double *dRHS;
        cudaMalloc((void **) &dRHS, N * sizeof(double));

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        initializeRHS<<<numBlocks, blockSize>>>(dRHS, N);
        cudaMemcpy(RHS, dRHS, N * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < RHSVec->size(); i++) {
            RHSVec->data()[i] = RHS[i];
        }

        delete [] RHS;
        cudaFree(dRHS);
    }

}
