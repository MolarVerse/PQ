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

#include "mShake.hpp"

#include <format>   // for std::format

#include "constants.hpp"         // for constants
#include "mShakeReference.hpp"   // for MShakeReference
#include "mathUtilities.hpp"     // for dot
#include "matrix.hpp"            // for Matrix
#include "simulationBox.hpp"     // for SimulationBox
#include "timingsSettings.hpp"   // for settings

using namespace constraints;

/**
 * @brief init M - Shake
 *
 **/
void MShake::initMShake(simulationBox::SimulationBox &simBox)
{
    initMShakeReferences();
    initPosBeforeIntegration(simBox);
}

/**
 * @brief init M - Shake references
 *
 **/
void MShake::initMShakeReferences()
{
    for (auto &mShakeReference : _mShakeReferences)
    {
        const auto nAtoms = mShakeReference.getNumberOfAtoms();
        const auto nBonds = nAtoms * (nAtoms - 1) / 2;

        auto &atoms = mShakeReference.getAtoms();

        std::vector<double>           rSquaredRefs;
        linearAlgebra::Matrix<double> mShakeMatrix(nBonds, nBonds);

        size_t bond_ij = 0;
        for (size_t i = 0; i < nAtoms - 1; ++i)
        {
            atoms[i].initMass();
            const auto mass_i = atoms[i].getMass();

            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                atoms[j].initMass();
                const auto mass_j = atoms[j].getMass();

                const auto pos_i   = atoms[i].getPosition();
                const auto pos_j   = atoms[j].getPosition();
                const auto dxyz_ij = pos_i - pos_j;
                const auto r2_ij   = dot(dxyz_ij, dxyz_ij);

                rSquaredRefs.push_back(r2_ij);

                size_t bond_kl = 0;

                for (size_t k = 0; k < nAtoms - 1; ++k)
                {
                    for (size_t l = k + 1; l < nAtoms; ++l)
                    {
                        const auto pos_k   = atoms[k].getPosition();
                        const auto pos_l   = atoms[l].getPosition();
                        const auto dxyz_kl = pos_k - pos_l;

                        const auto mShakeElement = calculateShakeMatrixElement(
                            i,
                            j,
                            k,
                            l,
                            mass_i,
                            mass_j,
                            dxyz_ij,
                            dxyz_kl
                        );

                        mShakeMatrix(bond_ij, bond_kl) = mShakeElement;

                        ++bond_kl;
                    }
                }

                ++bond_ij;
            }
        }

        _mShakeRSquaredRefs.push_back(rSquaredRefs);
        _mShakeMatrices.push_back(mShakeMatrix);

        const auto invMatrix = mShakeMatrix.inverse();

        _mShakeInvMatrices.push_back(invMatrix);
    }
}

/**
 * @brief init positions before integration
 *
 * @param simBox
 *
 */
void MShake::initPosBeforeIntegration(simulationBox::SimulationBox &simBox)
{
    const auto &molecules = simBox.getMolecules();

    for (const auto &molecule : molecules)
        _posBeforeIntegration.push_back(molecule.getAtomPositions());
}

/**
 * @brief applies the mShake algorithm to all bond constraints
 *
 * @param simulationBox
 *
 */
void MShake::applyMShake(
    const double                  rattleTolerance,
    simulationBox::SimulationBox &simulationBox
)
{
    auto &molecules = simulationBox.getMolecules();

    const auto dt          = settings::TimingsSettings::getTimeStep();
    const auto timeFactor  = 4.0 * dt * dt;
    const auto shakeFactor = 2.0 * dt * dt;

    for (size_t mol = 0; mol < molecules.size(); ++mol)
    {
        auto      &molecule = molecules[mol];
        const auto moltype  = molecule.getMoltype();

        if (!isMShakeType(moltype))
            continue;

        const auto mShakeIndex  = findMShakeReferenceIndex(moltype);
        const auto mShakeR2Refs = _mShakeRSquaredRefs[mShakeIndex];
        const auto nAtoms       = molecule.getNumberOfAtoms();
        const auto nBonds       = _mShakeInvMatrices[mShakeIndex].rows();

        auto &atoms            = molecule.getAtoms();
        auto  posUnconstrained = _posBeforeIntegration[mol];

        std::vector<linearAlgebra::Vec3D> bondsUnconstrained(nBonds);
        std::vector<double>               shakeVector(nBonds);
        linearAlgebra::Matrix<double>     mShakeMatrix(nBonds, nBonds);

        size_t index_ij = 0;

        for (size_t i = 0; i < nAtoms - 1; ++i)
            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                const auto pos_i = _posBeforeIntegration[mol][i];
                const auto pos_j = _posBeforeIntegration[mol][j];
                auto       dxyz  = pos_i - pos_j;

                simulationBox.applyPBC(dxyz);

                const auto r2    = dot(dxyz, dxyz);
                const auto r2Ref = mShakeR2Refs[index_ij];

                const auto r2Deviation = r2 - r2Ref;

                bondsUnconstrained[index_ij] = dxyz;
                shakeVector[index_ij]        = r2Deviation / timeFactor;

                ++index_ij;
            }

        while (true)
        {
            auto converged = true;
            index_ij       = 0;

            for (size_t i = 0; i < nAtoms - 1; ++i)
            {
                const auto mass_i = atoms[i]->getMass();

                for (size_t j = i + 1; j < nAtoms; ++j)
                {
                    const auto mass_j   = atoms[j]->getMass();
                    size_t     index_kl = 0;

                    for (size_t k = 0; k < nAtoms - 1; ++k)
                    {
                        for (size_t l = 0; l < nAtoms - 1; ++l)
                        {
                            const auto mShakeElement =
                                calculateShakeMatrixElement(
                                    i,
                                    j,
                                    k,
                                    l,
                                    mass_i,
                                    mass_j,
                                    bondsUnconstrained[index_ij],
                                    bondsUnconstrained[index_kl]
                                );

                            mShakeMatrix(index_ij, index_kl) = mShakeElement;

                            ++index_kl;
                        }
                    }

                    ++index_ij;
                }
            }

            const auto solutionVector = mShakeMatrix.solve(shakeVector);

            index_ij = 0;

            for (size_t i = 0; i < nAtoms - 1; ++i)
            {
                const auto mass_i = atoms[i]->getMass();

                for (size_t j = i + 1; j < nAtoms; ++j)
                {
                    const auto mass_j = atoms[j]->getMass();

                    auto posAdjustment = shakeFactor *
                                         solutionVector[index_ij] *
                                         _posBeforeIntegration[mol][i];

                    posUnconstrained[i] -= posAdjustment / mass_i;
                    posUnconstrained[j] += posAdjustment / mass_j;

                    atoms[i]->addVelocity(-posAdjustment / (mass_i * dt));
                    atoms[j]->addVelocity(posAdjustment / (mass_j * dt));

                    ++index_ij;
                }
            }

            index_ij = 0;

            for (size_t i = 0; i < nAtoms - 1; ++i)
            {
                for (size_t j = i + 1; j < nAtoms; ++j)
                {
                    auto pos = posUnconstrained[i] - posUnconstrained[j];
                    simulationBox.applyPBC(pos);

                    const auto r2    = dot(pos, pos);
                    const auto r2Ref = mShakeR2Refs[index_ij];

                    const auto r2Deviation = r2 - r2Ref;

                    shakeVector[index_ij]        = r2Deviation / timeFactor;
                    bondsUnconstrained[index_ij] = pos;

                    if (::abs(r2Deviation) / (2.0 * r2Ref) > rattleTolerance)
                        converged = false;

                    ++index_ij;
                }
            }

            if (!converged)
                break;
        }

        for (size_t i = 0; i < nAtoms; ++i)
            atoms[i]->setPosition(posUnconstrained[i]);

        molecule.calculateCenterOfMass(simulationBox.getBox());
    }
}

void MShake::applyMRattle(
    const double                  rattleTolerance,
    simulationBox::SimulationBox &simulationBox
)
{
    auto &molecules = simulationBox.getMolecules();

    for (size_t mol = 0; mol < molecules.size(); ++mol)
    {
        auto      &molecule = molecules[mol];
        const auto moltype  = molecule.getMoltype();

        if (!isMShakeType(moltype))
            continue;

        const auto mShakeIndex  = findMShakeReferenceIndex(moltype);
        const auto mShakeR2Refs = _mShakeRSquaredRefs[mShakeIndex];
        const auto nAtoms       = molecule.getNumberOfAtoms();
        const auto nBonds       = _mShakeInvMatrices[mShakeIndex].rows();

        auto &atoms = molecule.getAtoms();

        std::vector<double>               rattleVector(nBonds);
        std::vector<linearAlgebra::Vec3D> bonds(nBonds);

        size_t index_ij = 0;
        for (size_t i = 0; i < nAtoms - 1; ++i)
        {
            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                const auto pos_i = atoms[i]->getPosition();
                const auto pos_j = atoms[j]->getPosition();
                auto       dxyz  = pos_i - pos_j;

                simulationBox.applyPBC(dxyz);

                const auto v_i = atoms[i]->getVelocity();
                const auto v_j = atoms[j]->getVelocity();
                const auto dv  = v_i - v_j;

                bonds[index_ij]        = dxyz;
                rattleVector[index_ij] = dot(dxyz, dv);

                ++index_ij;
            }
        }

        index_ij = 0;
        for (size_t i = 0; i < nAtoms - 1; ++i)
        {
            const auto mass_i = atoms[i]->getMass();

            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                const auto mass_j = atoms[j]->getMass();

                auto velConstraint = 0.0;

                for (size_t k = 0; k < nBonds; ++k)
                {
                    velConstraint += _mShakeMatrices[mShakeIndex](index_ij, k) *
                                     rattleVector[k];
                }

                const auto velAdjustment = velConstraint * bonds[index_ij];

                atoms[i]->addVelocity(velAdjustment / mass_i);
                atoms[j]->addVelocity(-velAdjustment / mass_j);

                ++index_ij;
            }
        }
    }
}

/**
 * @brief check if molecule type is M - Shake type
 *
 * @param moltype
 *
 * @return bool
 */
bool MShake::isMShakeType(const size_t moltype) const
{
    bool isMShake = false;

    for (const auto &mShakeReference : _mShakeReferences)
    {
        const auto &moleculeType = mShakeReference.getMoleculeType();

        if (moleculeType.getMoltype() == moltype)
        {
            isMShake = true;
            break;
        }
    }

    return isMShake;
}

/**
 * @brief find M - Shake reference by molecule type
 *
 * @param moltype
 *
 * @return bool
 *
 * @throw customException::MShakeException if no M - Shake reference is
 * found
 */
const MShakeReference &MShake::findMShakeReference(const size_t moltype) const
{
    for (const auto &mShakeReference : _mShakeReferences)
    {
        const auto &moleculeType = mShakeReference.getMoleculeType();

        if (moleculeType.getMoltype() == moltype)
        {
            return mShakeReference;
        }
    }

    throw customException::MShakeException(
        std::format("No M-Shake reference found for molecule type {}", moltype)
    );
}

/**
 * @brief find M - Shake reference index by molecule type
 *
 * @param moltype
 *
 * @return size_t
 *
 * @throw customException::MShakeException if no M - Shake reference is
 * found
 */
size_t MShake::findMShakeReferenceIndex(const size_t moltype) const
{
    size_t index = 0;

    for (const auto &mShakeReference : _mShakeReferences)
    {
        const auto &moleculeType = mShakeReference.getMoleculeType();

        if (moleculeType.getMoltype() == moltype)
        {
            return index;
        }

        ++index;
    }

    throw customException::MShakeException(
        std::format("No M-Shake reference found for molecule type {}", moltype)
    );
}

/**
 * @brief add M - Shake reference
 *
 **/
void MShake::addMShakeReference(const MShakeReference &mShakeReference)
{
    _mShakeReferences.push_back(mShakeReference);
}

/**
 * @brief get M - Shake references
 *
 * @return const std::vector<MShakeReference>&
 **/
const std::vector<MShakeReference> &MShake::getMShakeReferences() const
{
    return _mShakeReferences;
}

/**
 * @brief calculate number of bond constraints
 *
 * @param simulationBox
 *
 * @return size_t
 */
size_t MShake::calculateNumberOfBondConstraints(
    simulationBox::SimulationBox &simBox
) const
{
    size_t nBondConstraints = 0;

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto moltype = molecule.getMoltype();

        if (isMShakeType(moltype))
        {
            const auto mShakeIndex  = findMShakeReferenceIndex(moltype);
            nBondConstraints       += _mShakeRSquaredRefs[mShakeIndex].size();
        }
    }

    return nBondConstraints;
}

/**
 * @brief calculate M - Shake matrix element
 *
 */
double MShake::calculateShakeMatrixElement(
    const size_t               i,
    const size_t               j,
    const size_t               k,
    const size_t               l,
    const double               mass_i,
    const double               mass_j,
    const linearAlgebra::Vec3D pos_ij,
    const linearAlgebra::Vec3D pos_kl
)
{
    const auto ik = utilities::kroneckerDelta(i, k);
    const auto il = utilities::kroneckerDelta(i, l);
    const auto jk = utilities::kroneckerDelta(j, k);
    const auto jl = utilities::kroneckerDelta(j, l);

    auto mShakeElement  = (ik - il) / mass_i;
    mShakeElement      += (jl - jk) / mass_j;
    mShakeElement      *= dot(pos_ij, pos_kl);

    return mShakeElement;
}