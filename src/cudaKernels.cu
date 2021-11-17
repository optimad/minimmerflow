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








__global__ void dev_Mirco00_evalInterfaceValues
(
        std::size_t nInterfaces,
        std::size_t cellStride,
  const std::size_t *interfaceRawIds,
  const double      *interfaceCentroids,
  const std::size_t *cellRawIds, 
  const double      *cellValues,
        int         order, 
        double      *interfaceValues
)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];
    const double *interfaceCentroid = interfaceCentroids + 3 * interfaceRawId;

    // Cell information
    const std::size_t cellRawId = cellRawIds[i];
    const double *c_rho = cellValues + N_FIELDS * cellRawId;
    const double *c_mx  = c_rho + 1;
    const double *c_my  = c_mx  + 1;
    const double *c_mz  = c_my  + 1;
    const double *c_e   = c_mz  + 1;

    // Reconstruct interface values
    double* i_rho = interfaceValues + i;
    double* i_mx  = i_rho + cellStride;
    double* i_my  = i_mx  + cellStride;
    double* i_mz  = i_my  + cellStride;
    double* i_e   = i_mz  + cellStride;

    // Reconstruct the variables
    *i_rho = *c_rho;
    *i_mx  = *c_mx;
    *i_my  = *c_my;
    *i_mz  = *c_mz;
    *i_e   = *c_e;

    return;
};






__global__ void dev_Mirco00_UniformUpdateRHS
(
        std::size_t  nInterfaces,
        std::size_t  cellStride,
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
    double lk;
    double lun;
    double rk;
    double run;


    // Gas constants
    const double g  = 1.4;
    const double gm1 = 0.4;
    const double rgm1 = 2.5;
    const double gg = 0.4/2.0;


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
    const double* leftReconstruction  = leftReconstructions  + tid;
    const double* rightReconstruction = rightReconstructions + tid;

    /*
     *  + LEFT FLUX
     */
    const double lr  = *(leftReconstruction); // left rho
    const double lir = 1.0/lr;                // left specific volume
    const double lm1 = *(leftReconstruction + cellStride); // left momentum along x
    lk  = lm1*lm1;
    const double lm2 = *(leftReconstruction + 2*cellStride); // left momentum along y
    lk += lm2*lm2;
    const double lm3 = *(leftReconstruction + 3*cellStride); // left momentum along z
    lk += lm3*lm3;
    const double le  = *(leftReconstruction + 4*cellStride); // left total energy

    // compute left kinetic energy
    lk *= (lir*lir);

    // left primitive variables
    const double lT = (2*le*lir -lk)*gg; // left temperature
    const double lu = lm1*lir;           // left velocity along x
    lun = lu*nx;
    const double lv = lm2*lir;           // left velocity along y
    lun += lv*ny;
    const double lw = lm3*lir;           // left velocity along z
    lun += lw*nz;
    const double lp = lr*lT;             // left pressure

    // left speed of sound
    const double la  = sqrt( g*lT );
    const double leto = rgm1*lp + 0.5*lr*lk;

    // left mass flux
    const double lmf  = lr*lun;

    // left fluxes
    const double lf0 = lmf;
    const double lf1 = lmf * lu + lp*nx;
    const double lf2 = lmf * lv + lp*ny;
    const double lf3 = lmf * lw + lp*nz;
    const double lf4 = lun*(leto+lp);
    const double llambda = fabs(lun) + la;

    /*
     *  + RIGHT FLUX
     */
    const double rr  = *(rightReconstruction); // right rho
    const double rir = 1.0/rr;                 // right specific volume
    const double rm1 = *(rightReconstruction + cellStride); // right mumentum along x
    rk  = rm1*rm1;
    const  double rm2 = *(rightReconstruction + 2*cellStride); // right mumentum along y
    rk += rm2*rm2;
    const double rm3 = *(rightReconstruction + 3*cellStride); // right mumentum along z
    rk += rm3*rm3;
    const double re  = *(rightReconstruction + 4*cellStride); // right total energy

    // compute right kinetic energy
    rk *= (rir*rir);

    // right primitive variables
    const double rT = (2*re*rir -rk)*gg; // right temperature
    const double ru = rm1*rir;           // right velocity along x
    run = ru*nx;
    const double rv = rm2*rir;           // right velocity along y
    run += rv*ny;
    const double rw = rm3*rir;           // right velocity along z
    run += rw*nz;
    const double rp = rr*rT;             // right pressure


    // right speed of sound
    const double ra  = sqrt( g*rT );
    const double reto = rgm1*rp + 0.5*rr*rk;

    // right mass flux
    const double rmf  = rr*run;

    // right fluxes
    const double rf0 = rmf;
    const double rf1 = rmf * ru + rp*nx;
    const double rf2 = rmf * rv + rp*ny;
    const double rf3 = rmf * rw + rp*nz;
    const double rf4 = run*(reto+rp);

    const double rlambda = fabs(run) + ra;


    interfaceMaxEig = fmax( llambda, rlambda );

    interfaceFluxes[0] = 0.5*( rf0-lf0 ) - interfaceMaxEig*(rr -lr );
    interfaceFluxes[1] = 0.5*( rf1-lf1 ) - interfaceMaxEig*(rm1-lm1);
    interfaceFluxes[2] = 0.5*( rf2-lf2 ) - interfaceMaxEig*(rm2-lm2);
    interfaceFluxes[3] = 0.5*( rf3-lf3 ) - interfaceMaxEig*(rm3-lm3);
    interfaceFluxes[4] = 0.5*( rf4-lf4 ) - interfaceMaxEig*(re -le );


    /*
     *  + REDUCE MAXIMUM EIGENVALUE
     */
    dev_reduceMax_Mirco(interfaceMaxEig, nInterfaces, maxEig);

    /*
     *  + ACCUMULATE FLUXES ON CELLS
     */
    std::size_t leftCellRawId  = leftCellRawIds[tid];
    std::size_t rightCellRawId = rightCellRawIds[tid];

    double *leftRHS  = cellRHS + 5 * leftCellRawId;
    double *rightRHS = cellRHS + 5 * rightCellRawId;
    for (int k = 0; k < 5; ++k) {
      double interfaceContribution = interfaceArea * interfaceFluxes[k];
      atomicAdd(leftRHS + k,  - interfaceContribution);
      atomicAdd(rightRHS + k,   interfaceContribution);
    }
    // Exit point
    return;

};
