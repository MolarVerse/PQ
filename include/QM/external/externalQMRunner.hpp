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

#ifndef QM__EXTERNAL__EXTERNAL_QM_RUNNER_HPP
#define QM__EXTERNAL__EXTERNAL_QM_RUNNER_HPP

#include <mstd/types.hpp>
#include <string_view>

#include "qmRunner.hpp"

class ExternalQMRunnerTest;   // forward declaration

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class Box;             // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

class ExternalQMSettings;   // forward declaration
class FileSettings;         // forward declaration

namespace QM
{
    /**
     * @brief ExternalQMRunner inherits from QMRunner
     *
     */
    class ExternalQMRunner : public QMRunner
    {
       private:
        mstd::ConstRef<ExternalQMSettings> _externalQMSettings;

       public:
        explicit ExternalQMRunner(
            const QMSettings         &qmSettings,
            const ExternalQMSettings &externalQMSettings
        );
        ~ExternalQMRunner() override = default;

        void run(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData,
            molsys::Periodicity         per,
            const FileSettings         &fileSettings
        ) override;

        virtual void execute(molsys::SimulationBox &, const FileSettings &) = 0;

        /*******************************
         * standard getter and setters *
         *******************************/

       protected:
        virtual void _executeCommand(
            std::string_view command,
            std::string_view program
        ) const;

        [[nodiscard]]
        const ExternalQMSettings &_getExternalQMSettings() const;

        virtual void _writeCoordsFile(molsys::SimulationBox &) = 0;

        virtual void _writePointChargeFile(
            molsys::SimulationBox & /*simBox*/,
            const std::string & /*pointChargeFileName*/
        )
        {
        }

        virtual void _readStressTensor(
            molsys::Box & /*simBox*/,
            physicalData::PhysicalData & /*physData*/,
            const std::string & /*stressFileName*/
        )
        {
        }

       private:
        static void _readForceFile(
            molsys::SimulationBox      &simulationBox,
            physicalData::PhysicalData &physicalData,
            const QMSettings           &qmSettings,
            const std::string          &forceFileName
        );
        static void _readChargeFile(
            molsys::SimulationBox &simulationBox,
            const QMSettings      &qmSettings,
            const std::string     &chargeFileName
        );

        friend class ::ExternalQMRunnerTest;   // for testing private methods
    };
}   // namespace QM

#endif   // QM__EXTERNAL__EXTERNAL_QM_RUNNER_HPP
