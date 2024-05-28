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

#ifdef WITH_KOKKOS

/**
 * @brief get reference to KokkosSimulationBox
 *
 * @return simulationBox::KokkosSimulationBox&
 */
simulationBox::KokkosSimulationBox &Engine::getKokkosSimulationBox()
{
    return _kokkosSimulationBox;
}

/**
 * @brief get reference to KokkosLennardJones
 *
 * @return potential::KokkosLennardJones&
 */
potential::KokkosLennardJones &Engine::getKokkosLennardJones()
{
    return _kokkosLennardJones;
}

/**
 * @brief get reference to KokkosCoulombWolf
 *
 * @return potential::KokkosCoulombWolf&
 */
potential::KokkosCoulombWolf &Engine::getKokkosCoulombWolf()
{
    return _kokkosCoulombWolf;
}

/**
 * @brief get reference to KokkosPotential
 *
 * @return potential::KokkosPotential&
 */
potential::KokkosPotential &Engine::getKokkosPotential()
{
    return _kokkosPotential;
}

/**
 * @brief get reference to KokkosVelocityVerlet
 *
 * @return integrator::KokkosVelocityVerlet&
 */
integrator::KokkosVelocityVerlet &Engine::getKokkosVelocityVerlet()
{
    return _kokkosVelocityVerlet;
}

/**
 * @brief initialize KokkosSimulationBox
 *
 * @param numAtoms number of atoms
 */
void Engine::initKokkosSimulationBox(const size_t numAtoms)
{
    _kokkosSimulationBox = simulationBox::KokkosSimulationBox(numAtoms);
}

/**
 * @brief initialize KokkosLennardJones
 *
 * @param numAtomTypes number of atom types
 */
void Engine::initKokkosLennardJones(const size_t numAtomTypes)
{
    _kokkosLennardJones = potential::KokkosLennardJones(numAtomTypes);
}

/**
 * @brief initialize KokkosCoulombWolf
 *
 * @param coulombRadiusCutOff cutoff radius for coulomb interaction
 * @param kappa kappa parameter for coulomb interaction
 * @param wolfParameter1 parameter 1 for wolf method
 * @param wolfParameter2 parameter 2 for wolf method
 * @param wolfParameter3 parameter 3 for wolf method
 * @param prefactor prefactor for wolf method
 */
void Engine::initKokkosCoulombWolf(
    const double coulombRadiusCutOff,
    const double kappa,
    const double wolfParameter1,
    const double wolfParameter2,
    const double wolfParameter3,
    const double prefactor
)
{
    _kokkosCoulombWolf = potential::KokkosCoulombWolf(
        coulombRadiusCutOff,
        kappa,
        wolfParameter1,
        wolfParameter2,
        wolfParameter3,
        prefactor
    );
}

/**
 * @brief initialize KokkosPotential
 */
void Engine::initKokkosPotential()
{
    _kokkosPotential = potential::KokkosPotential();
}

/**
 * @brief initialize KokkosVelocityVerlet
 *
 * @param dt time step
 * @param velocityFactor factor for velocity
 * @param timeFactor factor for time
 */
void Engine::initKokkosVelocityVerlet(
    const double dt,
    const double velocityFactor,
    const double timeFactor
)
{
    _kokkosVelocityVerlet =
        integrator::KokkosVelocityVerlet(dt, velocityFactor, timeFactor);
}

#endif