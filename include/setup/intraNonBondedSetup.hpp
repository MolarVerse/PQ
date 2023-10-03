/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _INTRA_NON_BONDED_SETUP_HPP_

#define _INTRA_NON_BONDED_SETUP_HPP_

namespace engine
{
    class Engine;   // forward declaration
}

namespace setup
{
    void setupIntraNonBonded(engine::Engine &);

    /**
     * @class IntraNonBondedSetup
     *
     * @brief Setup intra non bonded interactions
     *
     */
    class IntraNonBondedSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit IntraNonBondedSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _INTRA_NON_BONDED_SETUP_HPP_