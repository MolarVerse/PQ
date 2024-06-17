/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _CUDA_LENNARD_JONES_PAIR_HPP_

#define _CUDA_LENNARD_JONES_PAIR_HPP_

#include "cuda_runtime.h"

#include "forceFieldNonCoulomb.hpp"   // for matrix_shared_pair
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "matrix.hpp"                 // for matrix

namespace potential
{

    /**
     * @brief Structure for the Lennard-Jones potential on the device
     */
    struct CudaLennardJones_t
    {
        size_t numAtomTypes;
        double* radialCutoffs;
        double* energyCutoffs;
        double* forceCutoffs;
        double* c6;
        double* c12;
    };   // struct CudaLennardJones_t

    /**
     * @class CudaLennardJones
     *
     * @brief containing all information about the Lennard-Jones potential
     */
    class CudaLennardJones
    {
    private:
        size_t _numAtomTypes;
        double* _radialCutoffs;
        double* _energyCutoffs;
        double* _forceCutoffs;
        double* _c6;
        double* _c12;

    public:
        CudaLennardJones(size_t numAtomTypes);

        CudaLennardJones() = default;
        ~CudaLennardJones() = default;

        void transferFromNonCoulombPairMatrix(matrix_shared_pair& pairMatrix);
        CudaLennardJones_t* getCudaLennardJones() const;
    };

    // calculate the Lennard-Jones potential
    __device__ void calculateLennardJonesKernel(
        CudaLennardJones_t* lennardJones,
        const double distance,
        double& force,
        const size_t vdWType_i,
        const size_t vdWType_j,
        double* nonCoulombEnergy
    );

}   // namespace potential

#endif   // _CUDA_LENNARD_JONES_PAIR_HPP_