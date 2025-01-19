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

#include "intraNonBonded.hpp"

#include <algorithm>    // for for_each
#include <format>       // for format
#include <functional>   // for identity
#include <ranges>       // for std::ranges::find_if
#include <string>       // for string

#include "debug.hpp"
#include "exceptions.hpp"
#include "simulationBox.hpp"

using namespace intraNonBonded;
using namespace potential;
using namespace customException;
using namespace simulationBox;
using namespace physicalData;

using std::ranges::find_if;

/**
 * @brief clones the IntraNonBonded object
 *
 * @return std::shared_ptr<IntraNonBonded>
 */
std::shared_ptr<IntraNonBonded> IntraNonBonded::clone() const
{
    return std::make_shared<IntraNonBonded>(*this);
}

/**
 * @brief find a intraNonBondedContainer by molType and return a pointer to it
 *
 * @param molType
 * @return IntraNonBondedContainer*
 */
IntraNonBondedContainer *IntraNonBonded::findIntraNonBondedContainerByMolType(
    const size_t molType
)
{
    auto findByMolType = [molType](const auto &intraNonBondedType)
    { return intraNonBondedType.getMolType() == molType; };

    const auto it = find_if(_intraNonBondedContainers, findByMolType);

    if (it != _intraNonBondedContainers.end())
        return std::to_address(it);
    else
        throw IntraNonBondedException(std::format(
            "IntraNonBondedContainer with molType {} not found!",
            molType
        ));
}

/**
 * @brief fill the _intraNonBondedMaps vector with IntraNonBondedMap objects
 *
 * @param box
 */
void IntraNonBonded::fillIntraNonBondedMaps(SimulationBox &box)
{
    auto fillSingleMap = [this](auto &molecule)
    {
        const auto molType = molecule.getMoltype();

        auto *intraNonBondedContainer =
            findIntraNonBondedContainerByMolType(molType);

        _intraNonBondedMaps.push_back(
            IntraNonBondedMap(&molecule, intraNonBondedContainer)
        );
    };

    std::ranges::for_each(box.getMolecules(), fillSingleMap);
}

/**
 * @brief calculate the intra non bonded interactions for each intraNonBondedMap
 *
 * @param box
 * @param physicalData
 */
void IntraNonBonded::calculate(SimulationBox &box, PhysicalData &physicalData)
{
    __DEBUG_INFO__("Calculating Inter Non bonded forces");

    startTimingsSection("IntraNonBonded");

    // TODO: implement this for device
    auto calculateSingleContr = [this, &box, &physicalData](auto &intraMap)
    {
        intraMap.calculate(
            _coulombPotential.get(),
            _nonCoulombPot.get(),
            box,
            physicalData
        );
    };

    box.flattenForces();
    box.flattenShiftForces();

    std::ranges::for_each(_intraNonBondedMaps, calculateSingleContr);

    stopTimingsSection("IntraNonBonded");
}

/*************************
 *                       *
 * standard add methods  *
 *                       *
 *************************/

/**
 * @brief add a IntraNonBondedContainer to the _intraNonBondedContainers vector
 *
 * @param type
 */
void IntraNonBonded::addIntraNonBondedContainer(
    const IntraNonBondedContainer &type
)
{
    _intraNonBondedContainers.push_back(type);
}

/**
 * @brief add a IntraNonBondedMap to the _intraNonBondedMaps vector
 *
 * @param interaction
 */
void IntraNonBonded::addIntraNonBondedMap(const IntraNonBondedMap &interaction)
{
    _intraNonBondedMaps.push_back(interaction);
}

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activate the IntraNonBonded object
 */
void IntraNonBonded::activate() { _isActivated = true; }

/**
 * @brief deactivate the IntraNonBonded object
 */
void IntraNonBonded::deactivate() { _isActivated = false; }

/**
 * @brief check if the IntraNonBonded object is active
 *
 * @return bool
 */
bool IntraNonBonded::isActive() const { return _isActivated; }

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set the nonCoulomb potential
 *
 * @param pot
 */
void IntraNonBonded::setNonCoulombPotential(
    const std::shared_ptr<NonCoulombPotential> &pot
)
{
    _nonCoulombPot = pot;
}

/**
 * @brief set the Coulomb potential
 *
 * @param pot
 */
void IntraNonBonded::setCoulombPotential(
    const std::shared_ptr<CoulombPotential> &pot
)
{
    _coulombPotential = pot;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the IntraNonBondedType
 *
 * @return IntraNonBondedType
 */
IntraNonBondedType IntraNonBonded::getIntraNonBondedType() const
{
    return _intraNonBondedType;
}

/**
 * @brief get the IntraNonBondedContainers
 *
 * @return vec_intra_container
 */
std::vector<IntraNonBondedContainer> IntraNonBonded::
    getIntraNonBondedContainers() const
{
    return _intraNonBondedContainers;
}

/**
 * @brief get the IntraNonBondedMaps
 *
 * @return std::vector<IntraNonBondedMap>
 */
std::vector<IntraNonBondedMap> IntraNonBonded::getIntraNonBondedMaps() const
{
    return _intraNonBondedMaps;
}