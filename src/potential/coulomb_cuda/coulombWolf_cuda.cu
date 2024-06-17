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

#include "coulombWolf_cuda.cuh"
#include "cuda_runtime.h"

using namespace potential;

/**
 * @brief Construct a new Coulomb Wolf:: Coulomb Wolf object
 *
 * @param coulombRadiusCutOff
 * @param kappa
    * @param wolfParameter1
    * @param wolfParameter2
    * @param wolfParameter3
    * @param prefactor
 */
CudaCoulombWolf::CudaCoulombWolf(
    const double coulombRadiusCutOff,
    const double kappa,
    const double wolfParameter1,
    const double wolfParameter2,
    const double wolfParameter3,
    const double prefactor
)
{
    _coulombRadiusCutOff = coulombRadiusCutOff;
    _kappa = kappa;
    _wolfParameter1 = wolfParameter1;
    _wolfParameter2 = wolfParameter2;
    _wolfParameter3 = wolfParameter3;
    _prefactor = prefactor;
}

CudaCoulombWolf_t* CudaCoulombWolf::getCudaCoulombWolf() const
{
    CudaCoulombWolf_t* coulombWolf = new CudaCoulombWolf_t;
    coulombWolf->coulombRadiusCutOff = _coulombRadiusCutOff;
    coulombWolf->kappa = _kappa;
    coulombWolf->wolfParameter1 = _wolfParameter1;
    coulombWolf->wolfParameter2 = _wolfParameter2;
    coulombWolf->wolfParameter3 = _wolfParameter3;
    coulombWolf->prefactor = _prefactor;
    return coulombWolf;
}

/**
 * @brief calculate the energy and force of the Coulomb potential with Wolf
 * summation as long range correction
 *
 * @link https://doi.org/10.1063/1.478738
 *
 * @param distance
 * @return std::pair<double, double>
 */
__device__ void calculateWolfKernel(
    CudaCoulombWolf_t* coulombWolf,
    const double distance,
    const double charge_i,
    const double charge_j,
    double& force,
    double* coulombEnergy
)
{
    double prefactor = coulombWolf->prefactor;
    double kappa = coulombWolf->kappa;
    double wolfParameter1 = coulombWolf->wolfParameter1;
    double wolfParameter2 = coulombWolf->wolfParameter2;
    double wolfParameter3 = coulombWolf->wolfParameter3;
    double rcCutOff = coulombWolf->coulombRadiusCutOff;

    double coulombPrefactor = charge_i * charge_j * prefactor;

    double kappaDistance = kappa * distance;
    double kappaDistanceSquared = kappaDistance * kappaDistance;
    double erfcFactor = erfc(kappaDistance);

    double energy = erfcFactor / distance - wolfParameter1;
    energy += wolfParameter3 * (distance - rcCutOff);

    double scalarForce = erfcFactor / (distance * distance);
    scalarForce -= wolfParameter3;
    scalarForce +=
        wolfParameter2 * exp(-kappaDistanceSquared) / distance;

    scalarForce *= coulombPrefactor;

    force += scalarForce;

    energy *= coulombPrefactor;

    // add energy to the coulombEnergy
    *coulombEnergy += energy;
    return;
}