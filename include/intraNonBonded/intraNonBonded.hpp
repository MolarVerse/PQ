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

#ifndef _INTRA_NON_BONDED_HPP_

#define _INTRA_NON_BONDED_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr
#include <vector>    // for vector

#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedMap.hpp"         // for IntraNonBondedMap
#include "timer.hpp"                     // for Timer

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace potential

namespace intraNonBonded
{
    using vec_intra_container = std::vector<IntraNonBondedContainer>;

    /**
     * @brief enum class for the different types of intra non bonded
     * interactions
     *
     */
    enum class IntraNonBondedType : size_t
    {
        NONE,
        GUFF,
        FORCE_FIELD
    };

    /**
     * @class IntraNonBonded
     *
     * @brief base class for intra non bonded interactions
     */
    class IntraNonBonded : public timings::Timer
    {
       protected:
        IntraNonBondedType _intraNonBondedType = IntraNonBondedType::NONE;
        bool               _isActivated        = false;

        std::shared_ptr<potential::NonCoulombPotential> _nonCoulombPotential;
        std::shared_ptr<potential::CoulombPotential>    _coulombPotential;
        std::vector<IntraNonBondedMap>                  _intraNonBondedMaps;
        vec_intra_container _intraNonBondedContainers;

       public:
        void calculate(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
        void fillIntraNonBondedMaps(simulationBox::SimulationBox &);

        [[nodiscard]] IntraNonBondedContainer *findIntraNonBondedContainerByMolType(
            const size_t
        );

        /*************************
         * standard add methods  *
         *************************/

        void addIntraNonBondedContainer(const IntraNonBondedContainer &type)
        {
            _intraNonBondedContainers.push_back(type);
        }
        void addIntraNonBondedMap(const IntraNonBondedMap &interaction)
        {
            _intraNonBondedMaps.push_back(interaction);
        }

        /*****************************
         * standard activate methods *
         *****************************/

        void               activate() { _isActivated = true; }
        void               deactivate() { _isActivated = false; }
        [[nodiscard]] bool isActive() const { return _isActivated; }

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPotential(
            const std::shared_ptr<potential::NonCoulombPotential> &pot
        )
        {
            _nonCoulombPotential = pot;
        }
        void setCoulombPotential(
            const std::shared_ptr<potential::CoulombPotential> &pot
        )
        {
            _coulombPotential = pot;
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] IntraNonBondedType getIntraNonBondedType() const
        {
            return _intraNonBondedType;
        }
        [[nodiscard]] vec_intra_container getIntraNonBondedContainers() const
        {
            return _intraNonBondedContainers;
        }
        [[nodiscard]] std::vector<IntraNonBondedMap> getIntraNonBondedMaps(
        ) const
        {
            return _intraNonBondedMaps;
        }
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_HPP_