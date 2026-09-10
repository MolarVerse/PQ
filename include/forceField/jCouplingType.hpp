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

#ifndef _J_COUPLING_TYPE_HPP_

#define _J_COUPLING_TYPE_HPP_

#include <cstddef>

#include "strongTypes.hpp"

namespace forceField
{
    class JCouplingType;   // forward declaration

    bool operator==(const JCouplingType &lhs, const JCouplingType &rhs);

    /**
     * @class JCouplingType
     *
     * @brief represents a j coupling type
     *
     * @details this is a class representing a j coupling type defined in the
     * parameter file
     *
     */
    class JCouplingType
    {
       private:
        size_t _id;
        bool   _upperSymmetry = true;
        bool   _lowerSymmetry = true;

        JCouplingParams _params;

       public:
        JCouplingType(const size_t id, const JCouplingParams &params);

        friend bool operator==(const JCouplingType &, const JCouplingType &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t                 getId() const;
        [[nodiscard]] const JCouplingParams &getParams() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setUpperSymmetry(const bool boolean);
        void setLowerSymmetry(const bool boolean);
    };

}   // namespace forceField

#endif   // _J_COUPLING_TYPE_HPP_
