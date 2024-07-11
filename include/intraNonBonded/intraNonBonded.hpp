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
#include "typeAliases.hpp"

namespace intraNonBonded
{
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

        std::shared_ptr<pq::NonCoulombPot> _nonCoulombPot;
        std::shared_ptr<pq::CoulombPot>    _coulombPotential;
        std::vector<IntraNonBondedMap>     _intraNonBondedMaps;

        std::vector<IntraNonBondedContainer> _intraNonBondedContainers;

       public:
        std::shared_ptr<IntraNonBonded> clone() const;

        void calculate(const pq::SimBox &, pq::PhysicalData &);
        void fillIntraNonBondedMaps(pq::SimBox &);

        [[nodiscard]] IntraNonBondedContainer *findIntraNonBondedContainerByMolType(
            const size_t
        );

        /*************************
         * standard add methods  *
         *************************/

        void addIntraNonBondedContainer(const IntraNonBondedContainer &type);
        void addIntraNonBondedMap(const IntraNonBondedMap &interaction);

        /*****************************
         * standard activate methods *
         *****************************/

        void               activate();
        void               deactivate();
        [[nodiscard]] bool isActive() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPotential(const pq::SharedNonCoulombPot &pot);
        void setCoulombPotential(const pq::SharedCoulombPot &pot);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] IntraNonBondedType getIntraNonBondedType() const;
        [[nodiscard]] std::vector<IntraNonBondedContainer> getIntraNonBondedContainers(
        ) const;
        [[nodiscard]] std::vector<IntraNonBondedMap> getIntraNonBondedMaps(
        ) const;
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_HPP_