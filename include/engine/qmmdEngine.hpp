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

#ifndef _QM_MD_ENGINE_HPP_

#define _QM_MD_ENGINE_HPP_

#include <memory>   // for unique_ptr

#include "mdEngine.hpp"          // for Engine
#include "qmCapableEngine.hpp"   // for QMCapableEngine
#include "qmRunner.hpp"          // for QMRunner
#include "qmRunnerManager.hpp"   // for QMRunnerManager
#include "qmSettings.hpp"        // for QMSettings

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
        ~QMMDEngine() override = default;

        void calculateForces() override;
    };

}   // namespace engine

#endif   // _QM_MD_ENGINE_HPP_