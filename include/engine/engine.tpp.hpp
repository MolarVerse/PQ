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

#ifndef _ENGINE_TPP_

#define _ENGINE_TPP_

#include "engine.hpp"

/***************************
 * make unique_ptr methods *
 ***************************/

namespace engine
{
    template <typename T>
    inline void Engine::makeIntegrator(T integrator)
    {
        _integrator = std::make_unique<T>(integrator);
    }
    template <typename T>
    inline void Engine::makePotential(T)
    {
        _potential = std::make_unique<T>();
    }
    template <typename T>
    inline void Engine::makeThermostat(T thermostat)
    {
        _thermostat = std::make_unique<T>(thermostat);
    }
    template <typename T>
    inline void Engine::makeManostat(T manostat)
    {
        _manostat = std::make_unique<T>(manostat);
    }
    template <typename T>
    inline void Engine::makeVirial(T virial)
    {
        _virial = std::make_unique<T>(virial);
    }
}   // namespace engine

#endif   // _ENGINE_TPP_