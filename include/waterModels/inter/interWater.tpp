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

#ifndef _INTER_WATER_TPP_

#define _INTER_WATER_TPP_

#include <cstdlib>

#include "atom.hpp"
#include "hybridSettings.hpp"
#include "interWater.hpp"
#include "potential.hpp"

namespace waterModel
{
    /**
     * @brief Return the partial charge for a tagged atom.
     *
     * @tparam T Charge tag type.
     */
    template <typename T>
    double InterWaterStrategy::getPartialCharge(molsys::Atom & /*atom*/) const
    {
        std::abort();
    }

    template <>
    inline double InterWaterStrategy::getPartialCharge<pot::QMChargeTag>(
        molsys::Atom &atom
    ) const
    {
        const auto useQMCharges = settings::HybridSettings::getUseQMCharges();

        if (atom.getQMCharge() && useQMCharges)
            return atom.getQMCharge().value();

        return atom.getPartialCharge();
    }

    template <>
    inline double InterWaterStrategy::getPartialCharge<pot::MMChargeTag>(
        molsys::Atom &atom
    ) const
    {
        return atom.getPartialCharge();
    }

}   // namespace waterModel

#endif   //  _INTER_WATER_TPP_
