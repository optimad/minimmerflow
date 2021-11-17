#ifndef __CUDA_KERNELS_H__
#define __CUDA_KERNELS_H__

__global__ void dev_Mirco00_UniformUpdateRHS
(
        std::size_t  nInterfaces,
        std::size_t cellStride,
  const std::size_t * __restrict__ interfaceRawIds,
  const double      * __restrict__ interfaceNormals,
  const double      * __restrict__ interfaceAreas,
  const std::size_t * __restrict__ leftCellRawIds,
  const std::size_t * __restrict__ rightCellRawIds,
  const double      * __restrict__ leftReconstructions,
  const double      * __restrict__ rightReconstructions,
        double      * __restrict__ cellRHS,
        double      * __restrict__ maxEig
);



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
);

#endif
