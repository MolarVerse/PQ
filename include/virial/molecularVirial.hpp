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

#ifndef _MOLECULAR_VIRIAL_HPP_

#define _MOLECULAR_VIRIAL_HPP_

#include "virial.hpp"

namespace virial
{
    /**
     * @class MolecularVirial
     *
     * @brief Class for virial calculation of molecular systems
     *
     * @details overrides calculateVirial() function to include intra-molecular
     * virial correction
     */
    class MolecularVirial : public Virial
    {
       public:
        MolecularVirial();

        std::shared_ptr<Virial> clone() const override;

        void calculateVirial(pq::SimBox &, pq::PhysicalData &) override;
        void intraMolecularVirialCorrection(pq::SimBox &, pq::PhysicalData &)
            override;
    };

}   // namespace virial

#endif   // _MOLECULAR_VIRIAL_HPP_