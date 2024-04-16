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

#ifndef _INTRA_NON_BONDED_CONTAINER_HPP_

#define _INTRA_NON_BONDED_CONTAINER_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

namespace intraNonBonded
{
    /**
     * @class IntraNonBondedContainer
     *
     * @brief represents a container for a single intra non bonded type
     */
    class IntraNonBondedContainer
    {
      private:
        size_t                        _molType;
        std::vector<std::vector<int>> _atomIndices;

      public:
        IntraNonBondedContainer(const size_t molType, const std::vector<std::vector<int>> &atomIndices)
            : _molType(molType), _atomIndices(atomIndices){};

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t                        getMolType() const { return _molType; }
        [[nodiscard]] std::vector<std::vector<int>> getAtomIndices() const { return _atomIndices; }
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_CONTAINER_HPP_
