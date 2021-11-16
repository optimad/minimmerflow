#include "cudaKernels.h"
#include "constants.hcu"

typedef unsigned long long int uint64_cu;


__device__ void dev_atomicMax_Mirco(const double value, double * const maxValue)
{
    if (*maxValue >= value) {
        return;
    }

    uint64_cu oldMaxValue = *((uint64_cu *) maxValue);
    uint64_cu assumedMaxValue;
    do {
        assumedMaxValue = oldMaxValue;
        if (__longlong_as_double(assumedMaxValue) >= value) {
            break;
        }

        oldMaxValue = atomicCAS((uint64_cu *) maxValue, assumedMaxValue, __double_as_longlong(value));
    } while (assumedMaxValue != oldMaxValue);
}

/**
 * Compute the maximum of double-precision floating point values.
 *
 * \param value is the value that is compared in order to determine the maximum
 * \param nElements is the number of the elements that will be compared
 * \param[in,out] maxValue is the address of the reference value which might
 * get updated with the maximum
 */
__device__ void dev_reduceMax_Mirco(const double value, const size_t nElements, double *maxValue)
{
    extern __shared__ double blockValues[];

    // Get thread and global ids
    int tid = threadIdx.x;
    int gid = (blockDim.x * blockIdx.x) + tid;

    // Put thread value in the array that stores block values
    blockValues[tid] = value;
    __syncthreads();

    // Evaluate the maximum of each block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && gid < nElements) {
            blockValues[tid] = max(blockValues[tid], blockValues[tid + s]);
        }
        __syncthreads();
    }

    // Evaluate the maximum among different blocks
    if (tid == 0) {
        dev_atomicMax_Mirco(blockValues[0], maxValue);
    }
}


__device__ __forceinline__ void devFlux
(
  const double* __restrict__ ww,
        double* __restrict__ ff,
        double* __restrict__ lambda,
        double g,
        double gm1,
        double rgm1,
        double gg,
        double nx,
        double ny,
        double nz
)
{
  double k;
  double un;
  const double r  = ww[DEV_FID_RHO];   // rho
  const double ir = 1.0/r;           // specific volume
  const double m1 = ww[DEV_FID_RHO_U]; // momentum along x
  k  = m1*m1;
  const double m2 = ww[DEV_FID_RHO_V]; // momentum along y
  k += m2*m2;
  const double m3 = ww[DEV_FID_RHO_W]; // momentum along z
  k += m3*m3;
  const double e  = ww[DEV_FID_RHO_E]; // total energy

  // compute left kinetic energy
  k *= (ir*ir);

  // left primitive variables
  const double T = (2*e*ir -k)*gg; // temperature
  const double u = m1*ir;           // velocity along x
  un = u*nx;
  const double v = m2*ir;           // velocity along y
  un += v*ny;
  const double w = m3*ir;           // velocity along z
  un += w*nz;
  const double p = r*T;             // pressure

  // left speed of sound
  const double a  = sqrt( g*T );
  const double eto = rgm1*p + 0.5*r*k;

  // left mass flux
  const double mf  = r*un;

  // left fluxes
  ff[DEV_FID_RHO]   = mf;
  ff[DEV_FID_RHO_U] = mf * u + p*nx;
  ff[DEV_FID_RHO_V] = mf * v + p*ny;
  ff[DEV_FID_RHO_W] = mf * w + p*nz;
  ff[DEV_FID_RHO_E] = un*(eto+p);
  *lambda = fabs(un) + a;

  // Exit point
  return;
};




__global__ void dev_Mirco00_UniformUpdateRHS
(
        std::size_t  nInterfaces,
  const std::size_t * __restrict__ interfaceRawIds,
  const double      * __restrict__ interfaceNormals,
  const double      * __restrict__ interfaceAreas,
  const std::size_t * __restrict__ leftCellRawIds,
  const std::size_t * __restrict__ rightCellRawIds,
  const double      * __restrict__ leftReconstructions,
  const double      * __restrict__ rightReconstructions,
        double      * __restrict__ cellRHS,
        double      * __restrict__ maxEig
)
{

    // Local variables
    int tid;
    int k;
    double interfaceFluxes[5];
    double interfaceMaxEig;

    double rk;
    double run;


    // Gas constants
    const double g  = DEV_GAMMA;
    const double gm1 = DEV_GAMMA-1.0;
    const double rgm1 = 1.0 / gm1;
    const double gg = gm1/2.0;


    // Get interface information
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= nInterfaces ) return;

    // indirect acces to the interface idx
    const std::size_t interfaceRawId = interfaceRawIds[tid];


    /*
     *  + GET DATA FROM THE GLOBAL MEMORY
     */
    const double* interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double  interfaceArea   = interfaceAreas[interfaceRawId];

    const double nx = interfaceNormal[0];
    const double ny = interfaceNormal[1];
    const double nz = interfaceNormal[2];

    // Evaluate the conservative fluxes
    const double* lw = leftReconstructions  + N_FIELDS * tid;
    const double* rw = rightReconstructions + N_FIELDS * tid;

    // Local arrays
    double lf[N_FIELDS]; // Left flux
    double rf[N_FIELDS]; // Right flux
    double llambda;      // Left  max eigenvalue
    double rlambda;      // Right max eigenvalue

    /*
     *  + Compute fluxes
     */
    devFlux( lw, lf, &llambda, g, gm1, rgm1, gg, nx, ny, nz );
    devFlux( rw, rf, &rlambda, g, gm1, rgm1, gg, nx, ny, nz );


    interfaceMaxEig = fmax( llambda, rlambda );

    for ( int i=0; i<N_FIELDS; i++ )
    {
      interfaceFluxes[0] = 0.5*( rf[i] - lf[i] ) - 
               interfaceMaxEig*( rw[i] - lw[i] );
    };


    /*
     *  + REDUCE MAXIMUM EIGENVALUE
     */
    dev_reduceMax_Mirco(interfaceMaxEig, nInterfaces, maxEig);

    /*
     *  + ACCUMULATE FLUXES ON CELLS
     */
    std::size_t leftCellRawId  = leftCellRawIds[tid];
    std::size_t rightCellRawId = rightCellRawIds[tid];

    double *leftRHS  = cellRHS + N_FIELDS * leftCellRawId;
    double *rightRHS = cellRHS + N_FIELDS * rightCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
      double interfaceContribution = interfaceArea * interfaceFluxes[k];
      atomicAdd(leftRHS + k,  - interfaceContribution);
      atomicAdd(rightRHS + k,   interfaceContribution);
    }
    // Exit point
    return;

};
