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

#include "engine.hpp"

using namespace engine;

#ifdef WITH_CUDA

/**
 * @brief get reference to CudaSimulationBox
 *
 * @return simulationBox::CudaSimulationBox&
 */
simulationBox::CudaSimulationBox &Engine::getCudaSimulationBox()
{
    return _cudaSimulationBox;
}

/**
 * @brief get reference to CudaLennardJones
 *
 * @return potential::CudaLennardJones&
 */
potential::CudaLennardJones &Engine::getCudaLennardJones()
{
    return _cudaLennardJones;
}

/**
 * @brief get reference to CudaCoulombWolf
 *
 * @return potential::CudaCoulombWolf&
 */
potential::CudaCoulombWolf &Engine::getCudaCoulombWolf()
{
    return _cudaCoulombWolf;
}

/**
 * @brief get reference to CudaPotential
 *
 * @return potential::CudaPotential&
 */
potential::CudaPotential &Engine::getCudaPotential()
{
    return _cudaPotential;
}

/**
 * @brief initialize CudaSimulationBox
 *
 * @param numAtoms number of atoms
 */
void Engine::initCudaSimulationBox(const size_t numAtoms)
{
    _cudaSimulationBox = simulationBox::CudaSimulationBox(numAtoms);
}

/**
 * @brief initialize CudaLennardJones
 *
 * @param numAtomTypes number of atom types
 */
void Engine::initCudaLennardJones(const size_t numAtomTypes)
{
    _cudaLennardJones = potential::CudaLennardJones(numAtomTypes);
}

/**
 * @brief initialize CudaCoulombWolf
 *
 * @param coulombRadiusCutOff cutoff radius for coulomb interaction
 * @param kappa kappa parameter for coulomb interaction
 * @param wolfParameter1 parameter 1 for wolf method
 * @param wolfParameter2 parameter 2 for wolf method
 * @param wolfParameter3 parameter 3 for wolf method
 * @param prefactor prefactor for wolf method
 */
void Engine::initCudaCoulombWolf(
    const double coulombRadiusCutOff,
    const double kappa,
    const double wolfParameter1,
    const double wolfParameter2,
    const double wolfParameter3,
    const double prefactor
)
{
    _cudaCoulombWolf = potential::CudaCoulombWolf(
        coulombRadiusCutOff,
        kappa,
        wolfParameter1,
        wolfParameter2,
        wolfParameter3,
        prefactor
    );
}

/**
 * @brief initialize CudaPotential
 */
void Engine::initCudaPotential()
{
    _cudaPotential = potential::CudaPotential();
}

#endif