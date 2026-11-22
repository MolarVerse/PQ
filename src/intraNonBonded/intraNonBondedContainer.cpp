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

#include "intraNonBondedContainer.hpp"

namespace intraNonBonded
{

    /**
     * @brief compares two partners by atom index and 1-4 flag
     *
     * @param other
     * @return bool
     */
    bool IntraNonBondedPartner::operator==(const IntraNonBondedPartner &other
    ) const
    {
        return atomIndex == other.atomIndex && isScaled14 == other.isScaled14;
    }

    /**
     * @brief constructor for IntraNonBondedContainer
     *
     * @param molType
     * @param partners partner atoms of each reference atom
     */
    IntraNonBondedContainer::IntraNonBondedContainer(
        MolType                                    molType,
        const std::vector<IntraNonBondedPartners> &partners
    )
        : _molType(molType), _partners(partners)
    {
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the molType
     *
     * @return MolType
     */
    MolType IntraNonBondedContainer::getMolType() const { return _molType; }

    /**
     * @brief get the partner atoms of each reference atom
     *
     * @return const std::vector<IntraNonBondedPartners>&
     */
    const std::vector<IntraNonBondedPartners> &IntraNonBondedContainer::
        getPartners() const
    {
        return _partners;
    }

}   // namespace intraNonBonded
