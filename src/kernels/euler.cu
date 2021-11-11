/*---------------------------------------------------------------------------*\
 *
 *  minimmerflow
 *
 *  Copyright (C) 2015-2021 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of minimmerflow.
 *
 *  minimmerflow is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  minimmerflow is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with minimmerflow. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#include "euler.hcu"
#include "reconstruction.hpp"
#include "utils_cuda.hpp"
#include <nvtx3/nvToolsExt.h>

#define uint64  unsigned long long

namespace euler {

double *devMaxEig;

/**
 * Compute the maximum of two double-precision floating point values using an
 * atomic operation.
 *
 * \param value is the value that is compared to the reference in order to
 * determine the maximum
 * \param[in,out] maxValue is the address of the reference value which might
 * get updated with the maximum
 */
__device__ void dev_atomicMax(const double value, double * const maxValue)
{
    if (*maxValue >= value) {
        return;
    }

    uint64 oldMaxValue = *((uint64 *) maxValue);
    uint64 assumedMaxValue;
    do {
        assumedMaxValue = oldMaxValue;
        if (__longlong_as_double(assumedMaxValue) >= value) {
            break;
        }

        oldMaxValue = atomicCAS((uint64 *) maxValue, assumedMaxValue, __double_as_longlong(value));
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
__device__ void dev_reduceMax(const double value, const size_t nElements, double *maxValue)
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
        dev_atomicMax(blockValues[0], maxValue);
    }
}

/*!
 * Calculates the conservative fluxes for a perfect gas.
 *
 * \param conservative is the conservative state
 * \param n is the normal direction
 * \param[out] fluxes on output will contain the conservative fluxes
 * \param[out] lambda on output will contain the maximum eigenvalue
 */
__device__ void dev_evalFluxes(const double *conservative, const double *n,
                               double *fluxes, double *lambda)
{
    // Compute variables
    double primitive[N_FIELDS];
    ::utils::dev_conservative2primitive(conservative, primitive);

    double u = primitive[DEV_FID_U];
    double v = primitive[DEV_FID_V];
    double w = primitive[DEV_FID_W];

    double vel2 = u * u + v * v + w * w;
    double un   = ::utils::dev_normalVelocity(primitive, n);
    double a    = std::sqrt(DEV_GAMMA * primitive[DEV_FID_T]);

    double p = primitive[DEV_FID_P];
    if (p < 0.) {
        printf("***** Negative pressure (%f) in flux computation!\n", p);
    }

    double rho = conservative[DEV_FID_RHO];
    if (rho < 0.) {
       printf("***** Negative density (%f) in flux computation!\n", rho);
    }

    double eto = p / (DEV_GAMMA - 1.) + 0.5 * rho * vel2;

    // Compute fluxes
    double massFlux = rho * un;

    fluxes[DEV_FID_EQ_C]   = massFlux;
    fluxes[DEV_FID_EQ_M_X] = massFlux * u + p * n[0];
    fluxes[DEV_FID_EQ_M_Y] = massFlux * v + p * n[1];
    fluxes[DEV_FID_EQ_M_Z] = massFlux * w + p * n[2];
    fluxes[DEV_FID_EQ_E]   = un * (eto + p);

    // Evaluate maximum eigenvalue
    *lambda = std::abs(un) + a;
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
__device__ void dev_evalSplitting(const double *conservativeL, const double *conservativeR, const double *n, double *fluxes, double *lambda)
{
    // Fluxes
    double fL[N_FIELDS];
    double lambdaL;
    dev_evalFluxes(conservativeL, n, fL, &lambdaL);

    double fR[N_FIELDS];
    double lambdaR;
    dev_evalFluxes(conservativeR, n, fR, &lambdaR);

    // Splitting
    *lambda = max(lambdaR, lambdaL);

    for (int k = 0; k < N_FIELDS; ++k) {
        fluxes[k] = 0.5 * ((fR[k] + fL[k]) - (*lambda) * (conservativeR[k] - conservativeL[k]));
    }
}

/*!
 * Update residual of cells associated with uniform interfaces.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceAreas are the areas of the interfaces
 * \param leftCellRawIds are the raw ids of the left cells
 * \param rightCellRawIds are the raw ids of the right cells
 * \param leftReconstructions are the left reconstructions
 * \param rightReconstructions are the right reconstructions
 * \param[out] cellRHS are the RHS of the cells
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_uniformUpdateRHS(std::size_t nInterfaces, const std::size_t *interfaceRawIds,
                                     const double *interfaceNormals, const double *interfaceAreas,
                                     const std::size_t *leftCellRawIds, const std::size_t *rightCellRawIds,
                                     const double *leftReconstructions, const double *rightReconstructions,
                                     double *cellRHS, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Info about the interface
    const double *interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double interfaceArea    = interfaceAreas[interfaceRawId];

    // Evaluate the conservative fluxes
    const double *leftReconstruction  = leftReconstructions  + N_FIELDS * i;
    const double *rightReconstruction = rightReconstructions + N_FIELDS * i;

    double interfaceFluxes[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(leftReconstruction, rightReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    // Update cell residuals
    std::size_t leftCellRawId  = leftCellRawIds[i];
    std::size_t rightCellRawId = rightCellRawIds[i];

    double *leftRHS  = cellRHS + N_FIELDS * leftCellRawId;
    double *rightRHS = cellRHS + N_FIELDS * rightCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
        double interfaceContribution = interfaceArea * interfaceFluxes[k];

        atomicAdd(leftRHS + k,  - interfaceContribution);
        atomicAdd(rightRHS + k,   interfaceContribution);
    }

    // Update maximum eigenvalue
    dev_reduceMax(interfaceMaxEig, nInterfaces, maxEig);
}



__global__ void dev_Mirco01_UniformUpdateRHS
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
        double      * __restrict__ maxEig,
        double      * __restrict__ Workspace 
)
{

    // Local variables
    int tid, k;

    double interfaceFluxes[N_FIELDS];
    double interfaceMaxEig;

    double nx;
    double ny;
    double nz;

    double lr;
    double lir;
    double lm1;
    double lm2;
    double lm3;
    double le;
    double lk;
    double lp;
    double lu;
    double lv;
    double lw;
    double lT;
    double leto;
    double lmf;
    double lun;
    double llambda;
    double la;
    double lf[5];

    double rr;
    double rir;
    double rm1;
    double rm2;
    double rm3;
    double re;
    double rk;
    double rp;
    double ru;
    double rv;
    double rw;
    double rT;
    double reto;
    double rmf;
    double run;
    double rlambda;
    double ra;
    double rf[5];

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
     *  Qui si passa da array di strutture a strutture di 
     *  array cambiando lo stride
     *  ========================================================================
     */
    // Info about the interface
    const double* interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double  interfaceArea   = interfaceAreas[interfaceRawId];

    nx = interfaceNormal[0];
    ny = interfaceNormal[1];
    nz = interfaceNormal[2];

    // Evaluate the conservative fluxes
    const double* leftReconstruction  = leftReconstructions  + 5 * tid;
    const double* rightReconstruction = rightReconstructions + 5 * tid;
    /*
     *  ========================================================================
     */

    // Local interface fluxes
    // interfaceFluxes[0] = 0.0;
    // interfaceFluxes[1] = 0.0;
    // interfaceFluxes[2] = 0.0;
    // interfaceFluxes[3] = 0.0;
    // interfaceFluxes[4] = 0.0;

    lr  = leftReconstruction[0]; // left rho
    lir = 1.0/lr;                // left specific volume
    lm1 = leftReconstruction[1]; // left momentum along x
    lm2 = leftReconstruction[2]; // left momentum along y
    lm3 = leftReconstruction[3]; // left momentum along z
    le  = leftReconstruction[4]; // left total energy

    // compute left kinetic energy
    lk = ( lm1*lm1 + lm2*lm2 + lm3*lm3);
    lk *= (lir*lir);

    // left primitive variables
    lT = (2*le*lir -lk)*gg; // left temperature
    lu = lm1*lir;           // left velocity along x
    lv = lm2*lir;           // left velocity along y
    lw = lm3*lir;           // left velocity along z
    lp = lr*lT;             // left pressure

    // left velocity projected to the face normal vector
    lun = nx*lu + ny*lv + nz*lw;

    // left speed of sound
    la  = sqrt( g*lT );

    leto = rgm1*lp + 0.5*lr*lk;

    // left mass flux
    lmf  = lr*lun;


    // left fluxes
    lf[0] = lmf;
    lf[1] = lmf * lu + lp*nx;
    lf[2] = lmf * lv + lp*ny;
    lf[3] = lmf * lw + lp*nz;
    lf[4] = lun*(leto+lp);

    llambda = fabs(lun) + la;



    // get conservative variables
    rr  = rightReconstruction[0]; // right rho
    rir = 1.0/rr;                 // right specific volume
    rm1 = rightReconstruction[1]; // right mumentum along x
    rm2 = rightReconstruction[2]; // right mumentum along y
    rm3 = rightReconstruction[3]; // right mumentum along z
    re  = rightReconstruction[4]; // right total energy

    // compute right kinetic energy
    rk = ( rm1*rm1 + rm2*rm2 + rm3*rm3); 
    rk *= (rir*rir);

    // right primitive variables
    rT = (2*re*rir -rk)*gg; // right temperature
    ru = rm1*rir;           // right velocity along x
    rv = rm2*rir;           // right velocity along y
    rw = rm3*rir;           // right velocity along z
    rp = rr*rT;             // right pressure

    // right velocity projected to the face normal vector
    run = nx*ru + ny*rv + nz*rw;

    // right speed of sound
    ra  = sqrt( g*rT );

    reto = rgm1*rp + 0.5*rr*rk;

    // right mass flux
    rmf  = rr*run;

    // right fluxes
    rf[0] = rmf;
    rf[1] = rmf * ru + rp*nx;
    rf[2] = rmf * rv + rp*ny;
    rf[3] = rmf * rw + rp*nz;
    rf[4] = run*(reto+rp);

    rlambda = fabs(run) + ra;

    interfaceMaxEig = fmax( llambda, rlambda );

    interfaceFluxes[0] = 0.5*( rf[0]-lf[0] ) - interfaceMaxEig*(rr -lr );
    interfaceFluxes[1] = 0.5*( rf[1]-lf[1] ) - interfaceMaxEig*(rm1-lm1);
    interfaceFluxes[2] = 0.5*( rf[2]-lf[2] ) - interfaceMaxEig*(rm2-lm2);
    interfaceFluxes[3] = 0.5*( rf[3]-lf[3] ) - interfaceMaxEig*(rm3-lm3);
    interfaceFluxes[4] = 0.5*( rf[4]-lf[4] ) - interfaceMaxEig*(re -le );



    /**
     * Ciclo di accumulo 
     */

    // Update maximum eigenvalue
    dev_reduceMax(interfaceMaxEig, nInterfaces, maxEig);

    // Update cell residuals
    std::size_t leftCellRawId  = leftCellRawIds[tid];
    std::size_t rightCellRawId = rightCellRawIds[tid];

    double *leftRHS  = cellRHS + 5 * leftCellRawId;
    double *rightRHS = cellRHS + 5 * rightCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
      double interfaceContribution = interfaceArea * interfaceFluxes[k];
      atomicAdd(leftRHS + k,  - interfaceContribution);
      atomicAdd(rightRHS + k,   interfaceContribution);
    }

    // Exit point
    return;

};










__global__ void dev_Mirco02_UniformUpdateRHS
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
        double      * __restrict__ maxEig,
        double      * __restrict__ Workspace
)
{

    // Local variables
    int tid, k;

    double interfaceFluxes[N_FIELDS];
    double interfaceMaxEig;

    // double nx;
    // double ny;
    // double nz;

    // double lr;
    // double lir;
    // double lm1;
    // double lm2;
    // double lm3;
    // double le;
    double lk;
    // double lp;
    // double lu;
    // double lv;
    // double lw;
    // double lT;
    // double leto;
    // double lmf;
    double lun;
    // double llambda;
    // double la;
    double lf[5];

    // double rr;
    // double rir;
    // double rm1;
    // double rm2;
    // double rm3;
    // double re;
    double rk;
    // double rp;
    // double ru;
    // double rv;
    // double rw;
    // double rT;
    // double reto;
    // double rmf;
    double run;
    // double rlambda;
    // double ra;
    double rf[5];

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
     *  Qui si passa da array di strutture a strutture di 
     *  array cambiando lo stride
     *  ========================================================================
     */
    // Info about the interface
    const double* interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double  interfaceArea   = interfaceAreas[interfaceRawId];

    const double nx = interfaceNormal[0];
    const double ny = interfaceNormal[1];
    const double nz = interfaceNormal[2];

    // Evaluate the conservative fluxes
    const double* leftReconstruction  = leftReconstructions  + 16 * tid;
    const double* rightReconstruction = rightReconstructions + 16 * tid;
    /*
     *  ========================================================================
     */

    // Local interface fluxes
    // interfaceFluxes[0] = 0.0;
    // interfaceFluxes[1] = 0.0;
    // interfaceFluxes[2] = 0.0;
    // interfaceFluxes[3] = 0.0;
    // interfaceFluxes[4] = 0.0;

    const double lr  = leftReconstruction[0]; // left rho
    const double lir = 1.0/lr;                // left specific volume
    const double lm1 = leftReconstruction[1]; // left momentum along x
    lk  = lm1*lm1;
    const double lm2 = leftReconstruction[2]; // left momentum along y
    lk += lm2*lm2;
    const double lm3 = leftReconstruction[3]; // left momentum along z
    lk += lm3*lm3;
    const double le  = leftReconstruction[4]; // left total energy

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
    lf[0] = lmf;
    lf[1] = lmf * lu + lp*nx;
    lf[2] = lmf * lv + lp*ny;
    lf[3] = lmf * lw + lp*nz;
    lf[4] = lun*(leto+lp);

    const double llambda = fabs(lun) + la;

    // get conservative variables
    const double rr  = rightReconstruction[0]; // right rho
    const double rir = 1.0/rr;                 // right specific volume
    const double rm1 = rightReconstruction[1]; // right mumentum along x
    rk  = rm1*rm1;
    const  double rm2 = rightReconstruction[2]; // right mumentum along y
    rk += rm2*rm2; 
    const double rm3 = rightReconstruction[3]; // right mumentum along z
    rk += rm3*rm3;
    const double re  = rightReconstruction[4]; // right total energy

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
    rf[0] = rmf;
    rf[1] = rmf * ru + rp*nx;
    rf[2] = rmf * rv + rp*ny;
    rf[3] = rmf * rw + rp*nz;
    rf[4] = run*(reto+rp);

    const double rlambda = fabs(run) + ra;

    interfaceMaxEig = fmax( llambda, rlambda );

    interfaceFluxes[0] = 0.5*( rf[0]-lf[0] ) - interfaceMaxEig*(rr -lr );
    interfaceFluxes[1] = 0.5*( rf[1]-lf[1] ) - interfaceMaxEig*(rm1-lm1);
    interfaceFluxes[2] = 0.5*( rf[2]-lf[2] ) - interfaceMaxEig*(rm2-lm2);
    interfaceFluxes[3] = 0.5*( rf[3]-lf[3] ) - interfaceMaxEig*(rm3-lm3);
    interfaceFluxes[4] = 0.5*( rf[4]-lf[4] ) - interfaceMaxEig*(re -le );



    /**
     * Ciclo di accumulo 
     */

    // Update maximum eigenvalue
    dev_reduceMax(interfaceMaxEig, nInterfaces, maxEig);

    // if ( Workspace != nullptr )
    // {
    //   double* wrk = Workspace  + 5 * tid;
    //  wrk[0] = interfaceFluxes[0];
    //  wrk[1] = interfaceFluxes[1];
    //  wrk[2] = interfaceFluxes[2];
    //  wrk[3] = interfaceFluxes[3];
    //  wrk[4] = interfaceFluxes[4];
    // }
    // else
    // {
      // Update cell residuals
      std::size_t leftCellRawId  = leftCellRawIds[tid];
      std::size_t rightCellRawId = rightCellRawIds[tid];

      double *leftRHS  = cellRHS + 5 * leftCellRawId;
      double *rightRHS = cellRHS + 5 * rightCellRawId;
      for (int k = 0; k < N_FIELDS; ++k) {
        double interfaceContribution = interfaceArea * interfaceFluxes[k];
        atomicAdd(leftRHS + k,  - interfaceContribution);
        atomicAdd(rightRHS + k,   interfaceContribution);
      }
    // }

    // Exit point
    return;

};







/*
      // Update cell residuals
      std::size_t leftCellRawId  = leftCellRawIds[tid];
      std::size_t rightCellRawId = rightCellRawIds[tid];

      double *leftRHS  = cellRHS + 5 * leftCellRawId;
      double *rightRHS = cellRHS + 5 * rightCellRawId;
      for (int k = 0; k < N_FIELDS; ++k) {
        double interfaceContribution = interfaceArea * interfaceFluxes[k];
        atomicAdd(leftRHS + k,  - interfaceContribution);
        atomicAdd(rightRHS + k,   interfaceContribution);
      }
*/










/*!
 * Update residual of cells associated with boundary interfaces.
 *
 * \param nInterfaces is the number of solved interfaces
 * \param interfaceRawIds are the raw ids of the solved interfaces
 * \param interfaceNormals are the normals of the interfaces
 * \param interfaceAreas are the areas of the interfaces
 * \param fluidCellRawIds are the raw ids of the fluid cells
 * \param boundarySigns are the signs of the boundaries
 * \param fluidReconstructions are the fluid side reconstructions
 * \param virtualReconstructions are the virtual side reconstructions
 * \param[out] cellRHS are the RHS of the cells
 * \param[out] maxEig on output will containt the maximum eigenvalue
 */
__global__ void dev_boundaryUpdateRHS(std::size_t nInterfaces, const std::size_t *interfaceRawIds,
                                      const double *interfaceNormals, const double *interfaceAreas,
                                      const std::size_t *fluidCellRawIds, const std::size_t *boundarySigns,
                                      const double *fluidReconstructions, const double *virtualReconstructions,
                                      double *cellRHS, double *maxEig)
{
    // Get interface information
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nInterfaces) {
        return;
    }

    const std::size_t interfaceRawId = interfaceRawIds[i];

    // Info about the interface
    const double *interfaceNormal = interfaceNormals + 3 * interfaceRawId;
    const double interfaceArea    = interfaceAreas[interfaceRawId];

    // Info abount the bounday
    const int boundarySign = boundarySigns[i];

    // Evaluate the conservative fluxes
    const double *fluidReconstruction   = fluidReconstructions   + N_FIELDS * i;
    const double *virtualReconstruction = virtualReconstructions + N_FIELDS * i;

    double interfaceFluxes[N_FIELDS];
    for (int k = 0; k < N_FIELDS; ++k) {
        interfaceFluxes[k] = 0.;
    }

    double interfaceMaxEig;

    dev_evalSplitting(fluidReconstruction, virtualReconstruction, interfaceNormal, interfaceFluxes, &interfaceMaxEig);

    // Update residual of fluid cell
    std::size_t fluidCellRawId = fluidCellRawIds[i];
    double *fluidRHS = cellRHS + N_FIELDS * fluidCellRawId;
    for (int k = 0; k < N_FIELDS; ++k) {
        atomicAdd(fluidRHS + k, - boundarySign * interfaceArea * interfaceFluxes[k]);
    }

    // Update maximum eigenvalue
    dev_reduceMax(interfaceMaxEig, nInterfaces, maxEig);
}

/*!
 * Initialize CUDA computation
 */
void cuda_initialize()
{
    CUDA_ERROR_CHECK(cudaMalloc((void **) &devMaxEig, 1 * sizeof(double)));
}

/*!
 * Finalize CUDA computation
 */
void cuda_finalize()
{
    CUDA_ERROR_CHECK(cudaFree(devMaxEig));
}

/*!
 * Reset the RHS.
 *
 * \param[in,out] rhs is the RHS that will be reset
 */
void cuda_resetRHS(ScalarPiercedStorage<double> *cellsRHS)
{
    cellsRHS->cuda_fillDevice(0.);
}

/*!
 * Update cell RHS.
 *
 * \param problemType is the problem type
 * \param computationInfo are the computation information
 * \param order is the order
 * \param interfaceBCs is the boundary conditions storage
 * \param cellConservatives are the cell conservative values
 * \param[out] cellsRHS on output will containt the RHS
 * \param[out] maxEig on putput will containt the maximum eigenvalue
 */
void cuda_updateRHS(problem::ProblemType problemType, ComputationInfo &computationInfo,
                    const int order, const ScalarStorage<int> &solvedBoundaryInterfaceBCs,
                    const ScalarPiercedStorage<double> &cellConservatives, ScalarPiercedStorage<double> *cellsRHS, double *maxEig)
{
    //
    // Initialization
    //
    const double *devInterfaceNormals = computationInfo.cuda_getInterfaceNormalDevData();
    const double *devInterfaceAreas   = computationInfo.cuda_getInterfaceAreaDevData();

    CUDA_ERROR_CHECK(cudaMemset(devMaxEig, 0., 1 * sizeof(double)));

    double *devCellsRHS = cellsRHS->cuda_deviceData();

    const ScalarStorage<std::size_t> &solvedUniformInterfaceRawIds = computationInfo.getSolvedUniformInterfaceRawIds();
    const std::size_t nSolvedUniformInterfaces = solvedUniformInterfaceRawIds.size();

    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceRawIds = computationInfo.getSolvedBoundaryInterfaceRawIds();
    const std::size_t nSolvedBoundaryInterfaces = solvedBoundaryInterfaceRawIds.size();

    ScalarStorage<double> *leftReconstructions = &(computationInfo.getSolvedInterfaceLeftReconstructions());
    ScalarStorage<double> *rightReconstructions = &(computationInfo.getSolvedInterfaceRightReconstructions());

    double *devLeftReconstructions = computationInfo.getSolvedInterfaceLeftReconstructions().cuda_deviceData();
    double *devRightReconstructions = computationInfo.getSolvedInterfaceRightReconstructions().cuda_deviceData();

    //
    // Process uniform interfaces
    //
    const ScalarStorage<std::size_t> &solvedUniformInterfaceOwnerRawIds = computationInfo.getSolvedUniformInterfaceOwnerRawIds();
    const ScalarStorage<std::size_t> &solvedUniformInterfaceNeighRawIds = computationInfo.getSolvedUniformInterfaceNeighRawIds();

    // Reconstruct interface values
    nvtxRangePushA("InterfaceReconstructions");
    for (std::size_t i = 0; i < nSolvedUniformInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedUniformInterfaceRawIds[i];
        const std::array<double, 3> &interfaceCentroid = computationInfo.rawGetInterfaceCentroid(interfaceRawId);

        // Info about the interface owner
        std::size_t ownerRawId = solvedUniformInterfaceOwnerRawIds[i];
        const double *ownerMean = cellConservatives.rawData(ownerRawId);

        // Info about the interface neighbour
        std::size_t neighRawId = solvedUniformInterfaceNeighRawIds[i];
        const double *neighMean = cellConservatives.rawData(neighRawId);

        // Evaluate interface reconstructions
        double *ownerReconstruction = leftReconstructions->data() + 5 * i;
        double *neighReconstruction = rightReconstructions->data() + 5 * i;

        reconstruction::eval(ownerRawId, computationInfo, order, interfaceCentroid, ownerMean, ownerReconstruction);
        reconstruction::eval(neighRawId, computationInfo, order, interfaceCentroid, neighMean, neighReconstruction);
    }

    nvtxRangePop();
    nvtxRangePushA("UpdateInterfaceReconDev");
    leftReconstructions->cuda_updateDevice( 5 * nSolvedUniformInterfaces);
    rightReconstructions->cuda_updateDevice( 5 * nSolvedUniformInterfaces);
    nvtxRangePop();
    // Evaluate fluxes
    const std::size_t *devSolvedUniformInterfaceRawIds = solvedUniformInterfaceRawIds.cuda_deviceData();

    const std::size_t *devUniformOwnerRawIds = solvedUniformInterfaceOwnerRawIds.cuda_deviceData();
    const std::size_t *devUniformNeighRawIds = solvedUniformInterfaceNeighRawIds.cuda_deviceData();

    const int UNIFORM_BLOCK_SIZE = 256;
    int nUniformnBlocks = (nSolvedUniformInterfaces + UNIFORM_BLOCK_SIZE - 1) / UNIFORM_BLOCK_SIZE;
    int uniformSharedMemorySize = UNIFORM_BLOCK_SIZE * sizeof(double);
    /*
    dev_uniformUpdateRHS<<<nUniformnBlocks, UNIFORM_BLOCK_SIZE, uniformSharedMemorySize>>>(nSolvedUniformInterfaces, devSolvedUniformInterfaceRawIds,
                                                                                           devInterfaceNormals, devInterfaceAreas,
                                                                                           devUniformOwnerRawIds, devUniformNeighRawIds,
                                                                                           devLeftReconstructions, devRightReconstructions,
                                                                                           devCellsRHS, devMaxEig);
    */
    //double* m_deviceWorkspace;
    // cudaMalloc((void **) &m_deviceWorkspace, 12*2*nSolvedUniformInterfaces*sizeof(double)   );
    dev_Mirco02_UniformUpdateRHS<<<nUniformnBlocks, UNIFORM_BLOCK_SIZE, uniformSharedMemorySize>>>(nSolvedUniformInterfaces, devSolvedUniformInterfaceRawIds,
                                                                                           devInterfaceNormals, devInterfaceAreas,
                                                                                           devUniformOwnerRawIds, devUniformNeighRawIds,
                                                                                           devLeftReconstructions, devRightReconstructions,
                                                                                           devCellsRHS, devMaxEig, nullptr );
    //cudaFree( m_deviceWorkspace );

    /*
    //
    // Process boundary interfaces
    //
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceSigns = computationInfo.getSolvedBoundaryInterfaceSigns();
    const ScalarStorage<std::size_t> &solvedBoundaryInterfaceFluidRawIds = computationInfo.getSolvedBoundaryInterfaceFluidRawIds();

    // Reconstruct interface values
    nvtxRangePushA("BNDInterfaceReconstructions");
    for (std::size_t i = 0; i < nSolvedBoundaryInterfaces; ++i) {
        // Info about the interface
        const std::size_t interfaceRawId = solvedBoundaryInterfaceRawIds[i];
        const std::array<double, 3> &interfaceNormal = computationInfo.rawGetInterfaceNormal(interfaceRawId);
        const std::array<double, 3> &interfaceCentroid = computationInfo.rawGetInterfaceCentroid(interfaceRawId);
        int interfaceBC = solvedBoundaryInterfaceBCs[i];

        // Info about the interface fluid cell
        std::size_t fluidRawId = solvedBoundaryInterfaceFluidRawIds[i];
        const double *fluidMean = cellConservatives.rawData(fluidRawId);

        // Evaluate interface reconstructions
        double *fluidReconstruction   = leftReconstructions->data()  + N_FIELDS * i;
        double *virtualReconstruction = rightReconstructions->data() + N_FIELDS * i;

        reconstruction::eval(fluidRawId, computationInfo, order, interfaceCentroid, fluidMean, fluidReconstruction);
        evalInterfaceBCValues(problemType, interfaceBC, interfaceCentroid, interfaceNormal, fluidReconstruction, virtualReconstruction);
    }

    nvtxRangePop();
    nvtxRangePushA("BNDUpdateInterfaceReconDev");
    leftReconstructions->cuda_updateDevice( N_FIELDS * nSolvedBoundaryInterfaces);
    rightReconstructions->cuda_updateDevice( N_FIELDS * nSolvedBoundaryInterfaces);
    nvtxRangePop();

    // Evaluate fluxes
    const std::size_t *devSolvedBoundaryInterfaceRawIds = solvedBoundaryInterfaceRawIds.cuda_deviceData();

    const std::size_t *devBoundaryFluidRawIds = solvedBoundaryInterfaceFluidRawIds.cuda_deviceData();

    const std::size_t *devBoundarySigns = solvedBoundaryInterfaceSigns.cuda_deviceData();

    const int BOUNDARY_BLOCK_SIZE = 256;
    int nBoundarynBlocks = (nSolvedBoundaryInterfaces + BOUNDARY_BLOCK_SIZE - 1) / BOUNDARY_BLOCK_SIZE;
    int boundarySharedMemorySize = BOUNDARY_BLOCK_SIZE * sizeof(double);
    dev_boundaryUpdateRHS<<<nBoundarynBlocks, BOUNDARY_BLOCK_SIZE, boundarySharedMemorySize>>>(nSolvedBoundaryInterfaces, devSolvedBoundaryInterfaceRawIds,
                                                                                               devInterfaceNormals, devInterfaceAreas,
                                                                                               devBoundaryFluidRawIds, devBoundarySigns,
                                                                                               devLeftReconstructions, devRightReconstructions,
                                                                                               devCellsRHS, devMaxEig);
 
    */
    // Update host memory
    //
    cellsRHS->cuda_updateHost();
    CUDA_ERROR_CHECK(cudaMemcpy(maxEig, devMaxEig, sizeof(double), cudaMemcpyDeviceToHost));
}

}
