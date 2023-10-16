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

#ifndef _RING_POLYMER_QM_MD_ENGINE_HPP_

#define _RING_POLYMER_QM_MD_ENGINE_HPP_

#include "qmmdEngine.hpp"
#include "ringPolymerEngine.hpp"

namespace engine
{
    /**
     * @class RingPolymerQMMDEngine
     *
     * @details Contains all the information needed to run a ring polymer QM MD simulation
     *
     */
    class RingPolymerQMMDEngine : public QMMDEngine, public RingPolymerEngine
    {
      public:
        void takeStep() override;

        void qmCalculation();
    };
}   // namespace engine

#endif   // _RING_POLYMER_QM_MD_ENGINE_HPP_