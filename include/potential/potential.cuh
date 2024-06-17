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

# include "potential.hpp"

#include "timer.hpp"   // for Timer

namespace physicalData
{
    class PhysicalData;
}

namespace simulationBox
{
    class SimulationBox;
}   // namespace simulationBox


namespace potential {

    class LennardJones;   // forward declaration
    class CoulombWolf;    // forward declaration
    
    class PotentialCuda : public timings::Timer
    {
    public:
        // calculate forces
        void calculateForces(
            simulationBox::SimulationBox &simBox,
            simulationBox::simulationBoxCuda &simBoxCuda,
            physicalData::PhysicalData &physicalData
        );
    };
}   // namespace potential

# endif // POTENTIAL_CUDA_HPP