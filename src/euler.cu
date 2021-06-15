#include "storage.hpp"
#include "volume_kernel.hpp"
#include "patch_kernel.hpp"
#include "voloctree.hpp"
#include "utils.hpp"


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
    const double *n,
          double *fluxes,
    const int index,
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
)
{
    // Compute variables
    double u = primitive[index * N_FIELDS + FID_U];
    double v = primitive[index * N_FIELDS + FID_V];
    double w = primitive[index * N_FIELDS + FID_W];

    double vel2 = u * u + v * v + w * w;
    double un =
        primitive[index * N_FIELDS + FID_U] * n[index * 3 + 0]
      + primitive[index * N_FIELDS + FID_V] * n[index * 3 + 1]
      + primitive[index * N_FIELDS + FID_W] * n[index * 3 + 2];

    double p = primitive[index * N_FIELDS + FID_P];
//  if (p < 0.) {
//    log::cout() << "***** Negative pressure (" << p << ") in flux computation!\n";
//  }

    double rho = conservative[index * N_FIELDS + FID_RHO];
//  if (rho < 0.) {
//      log::cout() << "***** Negative density in flux computation!\n";
//  }

    double eto = p / (GAMMA - 1.) + 0.5 * rho * vel2;

    // Compute fluxes
    double massFlux = rho * un;

    fluxes[FID_EQ_C]   = massFlux;
    fluxes[FID_EQ_M_X] = massFlux * u + p * n[index * 3 + 0];
    fluxes[FID_EQ_M_Y] = massFlux * v + p * n[index * 3 + 1];
    fluxes[FID_EQ_M_Z] = massFlux * w + p * n[index * 3 + 2];
    fluxes[FID_EQ_E]   = un * (eto + p);
}



/*!
 * Computes approximate Riemann solver (Local Lax Friedrichs) for compressible perfect gas.
 *
 * \param conservativeL is the left conservative state
 * \param conservativeR is the right conservative state
 * \param n is the normal
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
//void evalSplitting(const double *conservativeL, const double *conservativeR, const std::array<double, 3> &n, FluxData *fluxes, double *lambda)
__global__
void evalSplitting_cu
(
    const double *primitiveL,
    const double *primitiveR,
    const double *conservativeL,
    const double *conservativeR,
    const double *n,
          double *fluxes,
          double *faceMaxEig,
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
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= Nif) {
        return;
    }
    // Fluxes
    double *fL = new double [N_FIELDS];
    double *fR = new double [N_FIELDS];

    evalFluxes_cu
    (
        conservativeL,
        primitiveL,
        n,
        fL,
        index,
        N_FIELDS,
        FID_U,
        FID_V,
        FID_W,
        FID_P,
        FID_RHO,
        FID_EQ_C,
        FID_EQ_M_X,
        FID_EQ_M_Y,
        FID_EQ_M_Z,
        FID_EQ_E,
        GAMMA
    );

    evalFluxes_cu
    (
        conservativeR,
        primitiveR,
        n,
        fR,
        index,
        N_FIELDS,
        FID_U,
        FID_V,
        FID_W,
        FID_P,
        FID_RHO,
        FID_EQ_C,
        FID_EQ_M_X,
        FID_EQ_M_Y,
        FID_EQ_M_Z,
        FID_EQ_E,
        GAMMA
    );

    // Eigenvalues
    double unL =
        primitiveL[index * N_FIELDS + FID_U] * n[index * 3 + 0]
      + primitiveL[index * N_FIELDS + FID_V] * n[index * 3 + 1]
      + primitiveL[index * N_FIELDS + FID_W] * n[index * 3 + 2];
    double aL = sqrt(GAMMA * primitiveL[index * N_FIELDS + FID_T]);
    double lambdaL = abs(unL) + aL;

    double unR =
        primitiveR[index * N_FIELDS + FID_U] * n[index * 3 + 0]
      + primitiveR[index * N_FIELDS + FID_V] * n[index * 3 + 1]
      + primitiveR[index * N_FIELDS + FID_W] * n[index * 3 + 2];
    double aR = sqrt(GAMMA * primitiveR[index * N_FIELDS + FID_T]);
    double lambdaR = abs(unR) + aR;

    double lambda = max(lambdaR, lambdaL);

    faceMaxEig[index] = lambda;

    // Splitting
    for (int k = 0; k < N_FIELDS; ++k) {
        fluxes[index * N_FIELDS + k] =
            0.5 * (
                    (fR[k] + fL[k])
                  - lambda
                  * (
                        conservativeR[index * N_FIELDS + k]
                      - conservativeL[index * N_FIELDS + k]
                    )
                 );
    }
    delete[] fL;
    delete[] fR;
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
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        RHS[index] = 0.;
    }
}


namespace CudaWrappers {

    void evalSplitting_wrapper
    (
        std::array<double, N_FIELDS> *conservativeOwner,
        std::array<double, N_FIELDS> *conservativeNeigh,
        std::array<double, 3> *interfaceNormal,
        double *fluxes,
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
    )
    {
        std::array<double, 5> *primitiveOwner
            = new std::array<double, 5>[Nif];
        std::array<double, 5> *primitiveNeigh
            = new std::array<double, 5>[Nif];
        for(int i = 0; i < Nif; i++){
            ::utils::conservative2primitive
            (
                conservativeOwner[i].data(),
                primitiveOwner[i].data()
            );
            ::utils::conservative2primitive
            (
                conservativeNeigh[i].data(),
                primitiveNeigh[i].data()
            );
        }

        std::vector<double> faceMaxEig(Nif);
        double *dprimitiveOwner;
        double *dprimitiveNeigh;
        double *dconservativeOwner;
        double *dconservativeNeigh;
        double *dfluxes;
        double *dinterfaceNormal;
        double *dfaceMaxEig;

        cudaMalloc((void **) &dprimitiveOwner,    Nif * N_FIELDS * sizeof(double));
        cudaMalloc((void **) &dprimitiveNeigh,    Nif * N_FIELDS * sizeof(double));
        cudaMalloc((void **) &dconservativeOwner, Nif * N_FIELDS * sizeof(double));
        cudaMalloc((void **) &dconservativeNeigh, Nif * N_FIELDS * sizeof(double));
        cudaMalloc((void **) &dfluxes,            Nif * N_FIELDS * sizeof(double));
        cudaMalloc((void **) &dfaceMaxEig,        Nif            * sizeof(double));
        cudaMalloc((void **) &dinterfaceNormal,   Nif * 3        * sizeof(double));

        cudaMemcpy(dprimitiveOwner,     primitiveOwner->data(),    Nif * N_FIELDS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dprimitiveNeigh,     primitiveNeigh->data(),    Nif * N_FIELDS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dconservativeOwner,  conservativeOwner->data(), Nif * N_FIELDS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dconservativeNeigh,  conservativeNeigh->data(), Nif * N_FIELDS * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dinterfaceNormal,    interfaceNormal->data(),   Nif * 3        * sizeof(double), cudaMemcpyHostToDevice);

        // TODO: Use a cuda function here to get appropriate blockSize
        int blockSize = 256;
        int numBlocks = (Nif + blockSize - 1) / blockSize;
        evalSplitting_cu<<<numBlocks, blockSize>>>
        (
            dprimitiveOwner,
            dprimitiveNeigh,
            dconservativeOwner,
            dconservativeNeigh,
            dinterfaceNormal,
            dfluxes,
            dfaceMaxEig,
            Nif,
            N_FIELDS,
            FID_U,
            FID_V,
            FID_W,
            FID_P,
            FID_RHO,
            FID_EQ_C,
            FID_EQ_M_X,
            FID_EQ_M_Y,
            FID_EQ_M_Z,
            FID_EQ_E,
            GAMMA
        );

        cudaMemcpy(fluxes,            dfluxes,     Nif * N_FIELDS * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(faceMaxEig.data(), dfaceMaxEig, Nif *            sizeof(double), cudaMemcpyDeviceToHost);

        *maxEig = 0.0;
        for(int i = 0; i < Nif; i++){
            *maxEig = std::max(faceMaxEig.data()[i], *maxEig);
        }

        cudaFree(dprimitiveOwner);
        cudaFree(dprimitiveNeigh);
        cudaFree(dconservativeOwner);
        cudaFree(dconservativeNeigh);
        cudaFree(dinterfaceNormal);
        cudaFree(dfaceMaxEig);
        cudaFree(dfluxes);
    }


    void initRHS_wrapper(double *RHS, const int N)
    {
        double *dRHS;
        cudaMalloc((void **) &dRHS, N * sizeof(double));

        int blockSize = 256;
        int numBlocks = (N + blockSize - 1) / blockSize;
        initializeRHS<<<numBlocks, blockSize>>>(dRHS, N);
        cudaMemcpy(RHS, dRHS, N * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(dRHS);
    }

}
