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

#ifndef QM__QM_RUNNER_FACTORY_HPP
#define QM__QM_RUNNER_FACTORY_HPP

#include <memory>

#include "qmRunner.hpp"

class QMSettings;

namespace QM
{
    /**
     * @class QMRunnerFactory
     *
     * @brief Factory class for creating QM runners
     *
     * @details This utility class provides static methods to create QM runners
     * based on the specified QM method. It encapsulates all the logic for
     * QM runner creation and configuration, making it reusable across different
     * engine types without requiring inheritance dependencies.
     */
    class QMRunnerFactory
    {
       public:
        [[nodiscard]]
        static std::unique_ptr<QMRunner> createQMRunner(
            const QMSettings& settings
        );

       private:
        QMRunnerFactory() = default;

        [[nodiscard]]
        static std::unique_ptr<QMRunner> _createAseDftbRunner(
            const QMSettings& settings
        );
        [[nodiscard]]
        static std::unique_ptr<QMRunner> _createAseFennolRunner(
            const QMSettings& settings
        );
        [[nodiscard]]
        static std::unique_ptr<QMRunner> _createAseMaceRunner(
            const QMSettings& settings
        );
        [[nodiscard]]
        static std::unique_ptr<QMRunner> _createAseXtbRunner(
            const QMSettings& settings
        );
    };

}   // namespace QM

#endif   // QM__QM_RUNNER_FACTORY_HPP
