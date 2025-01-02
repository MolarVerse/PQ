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

/**
 * @file potential.cpp
 * @author Jakob Gamper (97gamjak@gmail.com)
 * @brief This file contains the implementation of all member functions of the
 * Potential class that are not template functions and should not be inlined.
 *
 * @date 2024-12-09
 *
 */

#include "potential.hpp"

#include <cmath>   // for sqrt

#include "box.hpp"                       // for Box
#include "celllist.hpp"                  // for CellList
#include "coulombPotential.hpp"          // for CoulombPotential
#include "coulombShiftedPotential.hpp"   // for CoulombShiftedPotential
#include "coulombWolf.hpp"               // for CoulombWolf
#include "debug.hpp"                     // for debug
#include "forceFieldSettings.hpp"        // for ForceFieldSettings
#include "lennardJones.hpp"              // for LennardJones
#include "molecule.hpp"                  // for Molecule
#include "nonCoulombPair.hpp"            // for NonCoulombPair
#include "nonCoulombPotential.hpp"       // for NonCoulombPotential
#include "orthorhombicBox.hpp"           // for OrthorhombicBox
#include "physicalData.hpp"              // for PhysicalData
#include "potentialSettings.hpp"         // for PotentialSettings
#include "settings.hpp"                  // for Settings
#include "simulationBox.hpp"             // for SimulationBox

using namespace potential;
using namespace simulationBox;
using namespace settings;

/**
 * @brief calls the correct function pointer to a specialized template function
 * to calculate all inter non-bonded forces
 *
 * @param simBox
 * @param data
 * @param cellList
 */
void Potential::calculateForces(
    pq::SimBox&       simBox,
    pq::PhysicalData& data,
    pq::CellList&     cellList
)
{
    startTimingsSection("InterNonBonded");

    const auto rcCutOff = CoulombPotential::getCoulombRadiusCutOff();

    simBox.flattenForces();
    simBox.flattenShiftForces();
    simBox.flattenPositions();

    const size_t* atomtypes = nullptr;
    const size_t* molTypes  = nullptr;

    if (ForceFieldSettings::isActive())
        atomtypes = simBox.getInternalGlobalVDWTypesPtr();
    else
        atomtypes = simBox.getAtomTypesPtr();

    if (!ForceFieldSettings::isActive())
        molTypes = simBox.getMolTypesPtr();

    const auto* const moleculeIndex = simBox.getMoleculeIndicesPtr();
    const auto* const pos           = simBox.getPosPtr();
    const auto* const charge        = simBox.getChargesPtr();
    auto* const       force         = simBox.getForcesPtr();
    auto* const       shiftForce    = simBox.getShiftForcesPtr();

    simBox.getBox().updateBoxParams();   // TODO: remove this line later on
    const auto* const boxParams = simBox.getBox().getBoxParamsPtr();

    Real totalCoulombEnergy    = 0.0;
    Real totalNonCoulombEnergy = 0.0;

    if (cellList.isActive())
        throw customException::NotImplementedException(
            "The cell list is not implemented yet"
        );
    else
        _bruteForcePtr(
            pos,
            force,
            shiftForce,
            charge,
            getCoulParamsPtr(),
            getNonCoulParamsPtr(),
            getNonCoulCutOffsPtr(),
            boxParams,
            moleculeIndex,
            molTypes,
            atomtypes,
            totalCoulombEnergy,
            totalNonCoulombEnergy,
            rcCutOff,
            simBox.getNumberOfAtoms(),
            _nonCoulNumberOfTypes,
            _nonCoulParamsOffset,
            _maxNumAtomTypes,
            _numMolTypes
        );

#ifdef __PQ_DEBUG__
    // TODO: check why this always prints 0 for both energies
    if (config::Debug::useDebug(config::DebugLevel::ENERGY_DEBUG))
    {
        std::cout << std::format("Coulomb energy: {}\n", totalCoulombEnergy);
        std::cout << std::format(
            "Non-coulomb energy: {}\n",
            totalNonCoulombEnergy
        );
    }
#endif

    data.setCoulombEnergy(totalCoulombEnergy);
    data.setNonCoulombEnergy(totalNonCoulombEnergy);

    simBox.deFlattenForces();
    simBox.deFlattenShiftForces();

    stopTimingsSection("InterNonBonded");
}

/**
 * @brief sets the function pointers for the potential calculation
 *
 * @details The function pointers are set according to the settings
 * the user has chosen. The function pointers are used to store the
 * selected template function instantiation for the potential calculation.
 *
 * @param isBoxOrthogonal
 */
void Potential::setFunctionPointers(const bool isBoxOrthogonal)
{
    const auto nonCoulombType = PotentialSettings::getNonCoulombType();
    const auto coulombType    = PotentialSettings::getCoulombLongRangeType();

    const auto isFFActive = ForceFieldSettings::isActive();

    using enum NonCoulombType;
    using enum CoulombLongRangeType;

    const auto checkTypes = [&](const bool                 _isFFActive,
                                const NonCoulombType       _nonCoulombType,
                                const CoulombLongRangeType _coulombType,
                                const bool                 _isBoxOrthogonal)
    {
        if (isFFActive != _isFFActive)
            return false;

        if (nonCoulombType != _nonCoulombType)
            return false;

        if (coulombType != _coulombType)
            return false;

        if (isBoxOrthogonal != _isBoxOrthogonal)
            return false;

        return true;
    };

    if (checkTypes(true, LJ, SHIFTED, true))
    {
        // _cellListPtr = &potential::cellList<
        //     CoulombShiftedPotential,
        //     LennardJonesFF,
        //     OrthorhombicBox>;
        _bruteForcePtr = &potential::bruteForce<
            CoulombShiftedPotential,
            LennardJonesFF,
            OrthorhombicBox>;
    }
    else if (checkTypes(true, LJ, SHIFTED, false))
    {
        // _cellListPtr = &potential::cellList<
        //     CoulombShiftedPotential,
        //     LennardJonesFF,
        //     OrthorhombicBox>;
        _bruteForcePtr = &potential::bruteForce<
            CoulombShiftedPotential,
            LennardJonesFF,
            TriclinicBox>;
    }
    else if (checkTypes(true, LJ, WOLF, true))
    {
        // _cellListPtr = &potential::cellList<
        //     CoulombWolfPotential,
        //     LennardJonesFF,
        //     OrthorhombicBox>;
        _bruteForcePtr =
            &potential::
                bruteForce<CoulombWolf, LennardJonesFF, OrthorhombicBox>;
    }
    else if (checkTypes(true, LJ, WOLF, false))
    {
        // _cellListPtr = &potential::cellList<
        //     CoulombWolfPotential,
        //     LennardJonesFF,
        //     OrthorhombicBox>;
        _bruteForcePtr =
            &potential::bruteForce<CoulombWolf, LennardJonesFF, TriclinicBox>;
    }
    else
        throw customException::NotImplementedException(
            "The combination of the non-coulomb potential and the coulomb "
            "potential is not implemented yet"
        );
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the coulomb potential as a shared pointer
 *
 * @param pot
 */
void Potential::setNonCoulombPotential(
    const std::shared_ptr<NonCoulombPotential> pot
)
{
    _nonCoulombPot = pot;
}

/**
 * @brief set the non-coulomb potential parameters
 * and cutoffs
 *
 * @param params
 * @param cutOffs
 */
void Potential::setNonCoulombParamVectors(
    std::vector<Real> params,
    std::vector<Real> cutOffs,
    const size_t      nonCoulParamsOffset,
    const size_t      nonCoulNumberOfTypes
)
{
    _nonCoulParams        = std::move(params);
    _nonCoulCutOffs       = std::move(cutOffs);
    _nonCoulParamsOffset  = nonCoulParamsOffset;
    _nonCoulNumberOfTypes = nonCoulNumberOfTypes;
}

/**
 * @brief set the coulomb potential parameters
 *
 * @param coulParams
 */
void Potential::setCoulombParamVectors(std::vector<Real> coulParams)
{
    _coulParams = std::move(coulParams);
}

/**
 * @brief set maximum number of atom types
 *
 * @param maxNumAtomTypes
 */
void Potential::setMaxNumAtomTypes(const size_t maxNumAtomTypes)
{
    _maxNumAtomTypes = maxNumAtomTypes;
}

/**
 * @brief set the number of molecule types
 *
 * @param numMolTypes
 */
void Potential::setNumMolTypes(const size_t numMolTypes)
{
    _numMolTypes = numMolTypes;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the coulomb potential
 *
 * @return CoulombPotential&
 */
CoulombPotential& Potential::getCoulombPotential() const
{
    return *_coulombPotential;
}

/**
 * @brief get the non-coulomb potential
 *
 * @return NonCoulombPotential&
 */
NonCoulombPotential& Potential::getNonCoulombPotential() const
{
    return *_nonCoulombPot;
}

/**
 * @brief get the coulomb potential as a shared pointer
 *
 * @return SharedCoulombPot
 */
std::shared_ptr<CoulombPotential> Potential::getCoulombPotSharedPtr() const
{
    return _coulombPotential;
}

/**
 * @brief get the non-coulomb potential as a shared pointer
 *
 * @return SharedNonCoulombPot
 */
std::shared_ptr<NonCoulombPotential> Potential::getNonCoulombPotSharedPtr(
) const
{
    return _nonCoulombPot;
}

/**
 * @brief get the non-coulomb parameters pointer
 *
 * @return Real*
 */
Real* Potential::getNonCoulParamsPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _nonCoulParamsDevice;
    else
#endif
        return _nonCoulParams.data();
}

/**
 * @brief get the non-coulomb cutoffs pointer
 *
 * @return Real*
 */
Real* Potential::getNonCoulCutOffsPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _nonCoulCutOffsDevice;
    else
#endif
        return _nonCoulCutOffs.data();
}

/**
 * @brief get the coulomb parameters pointer
 *
 * @return Real*
 */
Real* Potential::getCoulParamsPtr()
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _coulParamsDevice;
    else
#endif
        return _coulParams.data();
}