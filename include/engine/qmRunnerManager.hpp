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

#ifndef _QM_RUNNER_MANAGER_HPP_

#define _QM_RUNNER_MANAGER_HPP_

#include <memory>

#include "qmRunner.hpp"
#include "qmSettings.hpp"

namespace engine
{
    /**
     * @class QMRunnerManager
     *
     * @brief Factory class for creating and managing QM runners
     *
     * @details This utility class provides static methods to create QM runners
     * based on the specified QM method. It encapsulates all the logic for
     * QM runner creation and configuration, making it reusable across different
     * engine types without requiring inheritance dependencies.
     */
    class QMRunnerManager
    {
       private:
        QMRunnerManager() = default;

       public:
        static std::shared_ptr<QM::QMRunner> createQMRunner(
            const settings::QMMethod method
        );
        static std::shared_ptr<QM::QMRunner> createMaceQMRunner();
        static std::shared_ptr<QM::QMRunner> createAseDftbRunner();
        static std::shared_ptr<QM::QMRunner> createAseXtbRunner();
    };

}   // namespace engine

#endif   // _QM_RUNNER_MANAGER_HPP_
