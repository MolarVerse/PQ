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

#ifndef _MAXWELL_BOLTZMANN_HPP_

#define _MAXWELL_BOLTZMANN_HPP_

#include <random>

#include "typeAliases.hpp"
#include "randomNumberGenerator.hpp"

using namespace randomNumberGenerator;

namespace maxwellBoltzmann
{
    /**
     * @class MaxwellBoltzmann
     *
     * @brief class to initialize velocities of particles with a random maxwell
     * boltzmann distribution
     *
     * @link https://www.biodiversitylibrary.org/item/53795#page/33/mode/1up
     * @link https://www.biodiversitylibrary.org/item/20012#page/37/mode/1up
     *
     */
    class MaxwellBoltzmann
    {
       private:
        RandomNumberGenerator _randomNumberGenerator{};

       public:
        void initializeVelocities(pq::SimBox &);
    };
}   // namespace maxwellBoltzmann

#endif   // _MAXWELL_BOLTZMANN_HPP_