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

#include "lennardJones_cuda.cuh"

using namespace potential;

/**
 * @brief constructor
 */
CudaLennardJones::CudaLennardJones(size_t numAtomTypes)
    : _numAtomTypes(numAtomTypes)
{
    // allocate memory
    cudaMallocManaged(&_radialCutoffs, numAtomTypes * numAtomTypes * sizeof(double));
    cudaMallocManaged(&_energyCutoffs, numAtomTypes * numAtomTypes * sizeof(double));
    cudaMallocManaged(&_forceCutoffs, numAtomTypes * numAtomTypes * sizeof(double));
    cudaMallocManaged(&_c6, numAtomTypes * numAtomTypes * sizeof(double));
    cudaMallocManaged(&_c12, numAtomTypes * numAtomTypes * sizeof(double));
}

/**
 * @brief transfer from non Coulomb pair matrix
 *
 * @param pairMatrix non Coulomb pair matrix
 */
void CudaLennardJones::transferFromNonCoulombPairMatrix(
    matrix_shared_pair& pairMatrix
)
{
    for (size_t i = 0; i < pairMatrix.rows(); ++i)
    {
        for (size_t j = 0; j < pairMatrix.cols(); ++j)
        {
            _radialCutoffs[i * _numAtomTypes + j] = pairMatrix(i, j)->getRadialCutOff();
            _energyCutoffs[i * _numAtomTypes + j] = pairMatrix(i, j)->getEnergyCutOff();
            _forceCutoffs[i * _numAtomTypes + j] = pairMatrix(i, j)->getForceCutOff();

            _c6[i * _numAtomTypes + j] =
                dynamic_cast<LennardJonesPair*>(pairMatrix(i, j).get())
                ->getC6();
            _c12[i * _numAtomTypes + j] =
                dynamic_cast<LennardJonesPair*>(pairMatrix(i, j).get())
                ->getC12();
        }
    }
}

CudaLennardJones_t* CudaLennardJones::getCudaLennardJones() const
{
    CudaLennardJones_t* lennardJones = new CudaLennardJones_t;
    lennardJones->numAtomTypes = _numAtomTypes;
    lennardJones->radialCutoffs = _radialCutoffs;
    lennardJones->energyCutoffs = _energyCutoffs;
    lennardJones->forceCutoffs = _forceCutoffs;
    lennardJones->c6 = _c6;
    lennardJones->c12 = _c12;
    return lennardJones;
}

/**
 * @brief Calculate the Lennard-Jones (12-6) energy and forces
 * between two atoms and add the forces to the force vector.
 *
 * @param lennardJones Lennard-Jones potential
 * @param distance distance between atoms
 * @param force force vector
 * @param vdWType_i van der Waals type of atom i
 * @param vdWType_j van der Waals type of atom j
 * @return energy of the Lennard-Jones potential
 */
__device__ __forceinline__ void calculateLennardJonesKernel(
    CudaLennardJones_t* lennardJones,
    const double distance,
    double& force,
    const size_t vdWType_i,
    const size_t vdWType_j,
    double* nonCoulombEnergy
)
{
    // calculate r^12 and r^6
    const auto distanceSquared = distance * distance;
    const auto distanceSixth =
        distanceSquared * distanceSquared * distanceSquared;
    const auto distanceTwelfth = distanceSixth * distanceSixth;

    const auto c12 = lennardJones->c12[vdWType_i * lennardJones->numAtomTypes + vdWType_j];
    const auto c6 = lennardJones->c6[vdWType_i * lennardJones->numAtomTypes + vdWType_j];
    const auto eCutoff = lennardJones->energyCutoffs[vdWType_i * lennardJones->numAtomTypes + vdWType_j];
    const auto fCutoff = lennardJones->forceCutoffs[vdWType_i * lennardJones->numAtomTypes + vdWType_j];
    const auto rCutoff = lennardJones->radialCutoffs[vdWType_i * lennardJones->numAtomTypes + vdWType_j];

    // calculate energy
    auto energy = c12 / distanceTwelfth;
    energy += c6 / distanceSixth;
    energy -= eCutoff;
    energy -= fCutoff * (rCutoff - distance);

    // calculate force
    auto scalarForce = 12.0 * c12 / (distanceTwelfth * distance);
    scalarForce += 6.0 * c6 / (distanceSixth * distance);
    scalarForce -= fCutoff;

    force += scalarForce;

    // add energy to non-Coulombic energy
    *nonCoulombEnergy += energy;
    return;
}