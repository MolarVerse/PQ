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
#include "distanceKernels.hpp"   // for distVecAndDist2
#include "mShakeReference.hpp"   // for MShakeReference
#include "mathUtilities.hpp"     // for dot
#include "matrix.hpp"            // for Matrix
#include "simulationBox.hpp"     // for SimulationBox
#include "stlVector.hpp"         // for dot
#include "timingsSettings.hpp"   // for settings

using namespace constraints;

/**
 * @brief init M - Shake
 *
 **/
void MShake::initMShake() { initMShakeReferences(); }

/**
 * @brief init M - Shake references
 *
 **/
void MShake::initMShakeReferences()
{
    for (auto &mShakeReference : _mShakeReferences)
    {
        auto &atoms = mShakeReference.getAtoms();

        /************************************
         * initialize the mass of all atoms *
         ************************************/
        for (auto &atom : atoms) atom.initMass();

        const auto nAtoms = mShakeReference.getNumberOfAtoms();
        const auto nBonds = nAtoms * (nAtoms - 1) / 2;

        std::vector<double>           rSquaredRefs;
        linearAlgebra::Matrix<double> mShakeMatrix(nBonds, nBonds);

        size_t bond_ij = 0;

        for (size_t i = 0; i < nAtoms - 1; ++i)
        {
            const auto mass_i = atoms[i].getMass();

            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                const auto mass_j = atoms[j].getMass();

                const auto [dxyz_ij, r2_ij] = kernel::distVecAndDist2(
                    atoms[i].getPosition(),
                    atoms[j].getPosition()
                );

                rSquaredRefs.push_back(r2_ij);

                size_t bond_kl = 0;
                for (size_t k = 0; k < nAtoms - 1; ++k)
                    for (size_t l = k + 1; l < nAtoms; ++l)
                    {
                        const auto dxyz_kl = kernel::distVec(
                            atoms[k].getPosition(),
                            atoms[l].getPosition()
                        );

                        const auto mShakeElement = calcMatrixElement(
                            {i, j, k, l},
                            {mass_i, mass_j},
                            {dxyz_ij, dxyz_kl}
                        );

                        mShakeMatrix(bond_ij, bond_kl) = mShakeElement;

                        ++bond_kl;
                    }

                ++bond_ij;
            }
        }

        /*********************
         * invert the matrix *
         *********************/
        const auto invMatrix = mShakeMatrix.inverse();

        /*******************************************************
         * add the calculated values to the respective vectors *
         * each molecule type has its own set of references    *
         * therefore, in the end each reference vector has a   *
         * size equal to the number of m-shake molecule types  *
         *******************************************************/
        _mShakeRSquaredRefs.push_back(rSquaredRefs);
        _mShakeMatrices.push_back(mShakeMatrix);
        _mShakeInvMatrices.push_back(invMatrix);
    }
}

/**
 * @brief applies the mShake algorithm to all bond constraints
 *
 * @param simulationBox
 *
 */
void MShake::applyMShake(
    const double                  shakeTolerance,
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
        auto      &atoms        = molecule.getAtoms();

        std::vector<linearAlgebra::Vec3D> bondsUnconstrained(nBonds);
        std::vector<linearAlgebra::Vec3D> bondsPrevious(nBonds);
        std::vector<double>               shakeVector(nBonds);
        linearAlgebra::Matrix<double>     mShakeMatrix(nBonds, nBonds);

        std::vector<linearAlgebra::Vec3D> posUnconstrained;

        /******************************************************
         * initialize the unconstrained positions of all atoms *
         *******************************************************/

        for (const auto &atom : atoms)
            posUnconstrained.push_back(atom->getPosition());

        /****************************************
         * initialize pre while loop iterations *
         ****************************************/

        size_t index_ij = 0;

        for (size_t i = 0; i < nAtoms - 1; ++i)
            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                /*************************************************
                 * determine bond vector of integrated positions *
                 *************************************************/

                const auto [dxyz, r2] = kernel::distVecAndDist2(
                    atoms[i]->getPosition(),
                    atoms[j]->getPosition(),
                    simulationBox
                );

                bondsUnconstrained[index_ij] = dxyz;

                /**************************************************
                 * shake vector from the deviation of the length  *
                 * of the bond vector from the reference value.   *
                 * Divide by the time factor to get a measurement *
                 * of a force.                                    *
                 **************************************************/

                const auto r2Ref = mShakeR2Refs[index_ij];

                const auto r2Deviation = r2 - r2Ref;

                shakeVector[index_ij] = r2Deviation / timeFactor;

                /*****************************************************
                 * determine bond vector of not integrated positions *
                 *****************************************************/

                const auto dxyz_prev = kernel::distVec(
                    atoms[i]->getPositionOld(),
                    atoms[j]->getPositionOld(),
                    simulationBox
                );

                bondsPrevious[index_ij] = dxyz_prev;

                ++index_ij;
            }

        while (true)
        {
            auto converged = true;
            index_ij       = 0;

            /*****************************************
             * fill (nBonds x nBonds) m-Shake matrix *
             *****************************************/

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
                            const auto mShakeElement = calcMatrixElement(
                                {i, j, k, l},
                                {mass_i, mass_j},
                                {bondsUnconstrained[index_ij],
                                 bondsUnconstrained[index_kl]}
                            );

                            mShakeMatrix(index_ij, index_kl) = mShakeElement;

                            ++index_kl;
                        }
                    }

                    ++index_ij;
                }
            }

            /*********************************************************
             * solve the linear system of equations                  *
             *        b = A * x                                      *
             * where b is the shakeVector, A is the mShakeMatrix and *
             * x is the solutionVector                               *
             *********************************************************/

            const auto solutionVector = mShakeMatrix.solve(shakeVector);

            index_ij = 0;

            /********************************************
             * adjust velocities and positions of atoms *
             ********************************************/

            for (size_t i = 0; i < nAtoms - 1; ++i)
            {
                const auto mass_i = atoms[i]->getMass();

                for (size_t j = i + 1; j < nAtoms; ++j)
                {
                    const auto mass_j = atoms[j]->getMass();

                    const auto &bondPrev = bondsPrevious[index_ij];
                    const auto &solution = solutionVector[index_ij];

                    auto posAdjustment  = solution * bondPrev;
                    posAdjustment      *= shakeFactor;

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
                    /*************************************************
                     * determine bond vector of integrated positions *
                     *************************************************/

                    const auto [dxyz, r2] = kernel::distVecAndDist2(
                        posUnconstrained[i],
                        posUnconstrained[j],
                        simulationBox
                    );

                    bondsUnconstrained[index_ij] = dxyz;

                    /**************************************************
                     * shake vector from the deviation of the length  *
                     * of the bond vector from the reference value.   *
                     * Divide by the time factor to get a measurement *
                     * of a force.                                    *
                     **************************************************/

                    const auto r2Ref       = mShakeR2Refs[index_ij];
                    const auto r2Deviation = r2 - r2Ref;
                    shakeVector[index_ij]  = r2Deviation / timeFactor;

                    /******************************************************
                     * check if the deviation of the bond length from the *
                     * reference value is larger than the tolerance value *
                     ******************************************************/

                    if (::abs(r2Deviation) / (2.0 * r2Ref) > shakeTolerance)
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

/**
 * @brief apply M - Rattle to correct velocities
 *
 * @param shakeTolerance
 * @param simulationBox
 *
 */
void MShake::applyMRattle(simulationBox::SimulationBox &simulationBox)
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
        auto       mShakeMatrix = _mShakeInvMatrices[mShakeIndex];
        const auto nAtoms       = molecule.getNumberOfAtoms();
        const auto nBonds       = mShakeMatrix.rows();

        auto &atoms = molecule.getAtoms();

        std::vector<double>               rattleVector(nBonds);
        std::vector<linearAlgebra::Vec3D> bonds(nBonds);

        size_t index_ij = 0;
        for (size_t i = 0; i < nAtoms - 1; ++i)
        {
            for (size_t j = i + 1; j < nAtoms; ++j)
            {
                const auto dxyz = kernel::distVec(
                    atoms[i]->getPosition(),
                    atoms[j]->getPosition(),
                    simulationBox
                );

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

                auto velConstraint = dot(mShakeMatrix(index_ij), rattleVector);

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
double MShake::calcMatrixElement(
    const std::tuple<size_t, size_t, size_t, size_t>            &indices,
    const std::pair<double, double>                             &masses,
    const std::pair<linearAlgebra::Vec3D, linearAlgebra::Vec3D> &pos
) const
{
    const auto i = std::get<0>(indices);
    const auto j = std::get<1>(indices);
    const auto k = std::get<2>(indices);
    const auto l = std::get<3>(indices);

    const auto ik = utilities::kroneckerDelta(i, k);
    const auto il = utilities::kroneckerDelta(i, l);
    const auto jk = utilities::kroneckerDelta(j, k);
    const auto jl = utilities::kroneckerDelta(j, l);

    const auto mass_i = masses.first;
    const auto mass_j = masses.second;

    const auto &pos_ij = pos.first;
    const auto &pos_kl = pos.second;

    auto mShakeElement  = (ik - il) / mass_i;
    mShakeElement      += (jl - jk) / mass_j;
    mShakeElement      *= dot(pos_ij, pos_kl);

    return mShakeElement;
}