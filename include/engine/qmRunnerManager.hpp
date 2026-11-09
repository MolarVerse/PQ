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

#ifndef ENGINE__QM_RUNNER_MANAGER_HPP
#define ENGINE__QM_RUNNER_MANAGER_HPP

#include <memory>

#include "qmRunner.hpp"

class QMSettings;
class MaceSettings;
class FennolSettings;
class DftbSettings;
class XtbSettings;

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
       public:
        [[nodiscard]]
        static std::shared_ptr<QM::QMRunner> createQMRunner(
            const QMSettings     &qmSettings,
            const MaceSettings   &maceSettings,
            const FennolSettings &fennolSettings,
            const DftbSettings   &dftbSettings,
            const XtbSettings    &xtbSettings
        );
        [[nodiscard]]
        static std::shared_ptr<QM::QMRunner> createAseDftbRunner(
            const QMSettings   &qmSettings,
            const DftbSettings &dftbSettings
        );
        [[nodiscard]]
        static std::shared_ptr<QM::QMRunner> createAseFennolRunner(
            const QMSettings     &qmSettings,
            const FennolSettings &fennolSettings
        );
        [[nodiscard]]
        static std::shared_ptr<QM::QMRunner> createAseMaceRunner(
            const QMSettings   &qmSettings,
            const MaceSettings &maceSettings
        );
        [[nodiscard]]
        static std::shared_ptr<QM::QMRunner> createAseXtbRunner(
            const QMSettings  &qmSettings,
            const XtbSettings &xtbSettings
        );

       private:
        QMRunnerManager() = default;
    };

}   // namespace engine

#endif   // ENGINE__QM_RUNNER_MANAGER_HPP
