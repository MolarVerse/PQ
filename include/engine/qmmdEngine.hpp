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

#ifndef ENGINE__QMMD_ENGINE_HPP
#define ENGINE__QMMD_ENGINE_HPP

#include "mdEngine.hpp"
#include "qmCapableEngine.hpp"

namespace engine
{

    /**
     * @class QMMDEngine
     *
     * @brief Contains all the information needed to run a QM MD simulation
     *
     */
    class QMMDEngine : virtual public MDEngine, public QMCapableEngine
    {
       public:
        using MDEngine::MDEngine;

        ~QMMDEngine() override = default;

        void calculateForces() override;
    };

}   // namespace engine

#endif   // ENGINE__QMMD_ENGINE_HPP
