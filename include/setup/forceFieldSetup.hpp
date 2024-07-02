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

#ifndef _FORCE_FIELD_SETUP_HPP_

#define _FORCE_FIELD_SETUP_HPP_

#include "typeAliases.hpp"

namespace setup
{
    void setupForceField(pq::Engine &);

    /**
     * @class ForceFieldSetup
     *
     * @brief setup all bonded contributions in the force field
     *
     */
    class ForceFieldSetup
    {
       private:
        pq::Engine &_engine;

        size_t _nBondTypes     = 0;
        size_t _nAngleTypes    = 0;
        size_t _nDihedralTypes = 0;
        size_t _nImproperTypes = 0;

       public:
        explicit ForceFieldSetup(pq::Engine &engine);

        void setup();
        void setupBonds();
        void setupAngles();
        void setupDihedrals();
        void setupImproperDihedrals();

        void writeSetupInfo();
    };

}   // namespace setup

#endif   // _FORCE_FIELD_SETUP_HPP_