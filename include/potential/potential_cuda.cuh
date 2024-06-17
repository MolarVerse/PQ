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

# ifndef POTENTIAL_CUDA_HPP
# define POTENTIAL_CUDA_HPP

# include "cuda_runtime.h"   // for cudaMallocManaged, cudaFree

# include "simulationBox.hpp"   // for SimulationBox
# include "simulationBox_cuda.cuh"   // for CudaSimulationBox
# include "lennardJones_cuda.cuh"   // for CudaLennardJones
# include "coulombWolf_cuda.cuh"    // for CudaCoulombWolf
# include "potential.hpp"
# include "physicalData.hpp"   // for PhysicalData

#include "timer.hpp"   // for Timer

using namespace simulationBox;
using namespace potential;
using namespace physicalData;

namespace potential {

    class CudaLennardJones;   // forward declaration
    class CudaCoulombWolf;    // forward declaration

    class CudaPotential : public timings::Timer
    {
    public:
        // calculate forces
        void calculateForces(
            SimulationBox& simBox,
            PhysicalData& physicalData,
            CudaSimulationBox& simBoxCuda,
            CudaLennardJones& lennardJones,
            CudaCoulombWolf& coulombWolf
        );
    };
}   // namespace potential

# endif // POTENTIAL_CUDA_HPP