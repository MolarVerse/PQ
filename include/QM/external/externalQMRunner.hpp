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

#ifndef _EXTERNAL_QM_RUNNER_HPP_

#define _EXTERNAL_QM_RUNNER_HPP_

#include <string>
#include <string_view>

#include "qmRunner.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class Box;             // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace QM
{
    [[nodiscard]] std::string bundledQMScriptPath(std::string_view script);

    /**
     * @brief ExternalQMRunner inherits from QMRunner
     *
     */
    class ExternalQMRunner : public QMRunner
    {
       protected:
        std::string            _scriptPath  = SCRIPT_PATH_;
        constexpr static auto *_singularity = SINGULARITY_;
        constexpr static auto *_staticBuild = STATIC_BUILD_;

        [[nodiscard]] std::string resolveScriptPath(
            std::string_view script
        ) const;

        virtual void executeCommand(
            std::string_view command,
            std::string_view program
        ) const;

       public:
        ExternalQMRunner()           = default;
        ~ExternalQMRunner() override = default;

        void run(
            molsys::SimulationBox &,
            physicalData::PhysicalData &,
            molsys::Periodicity per
        ) override;
        virtual void execute(molsys::SimulationBox &) = 0;

        virtual void writeCoordsFile(molsys::SimulationBox &) = 0;

        virtual void writePointChargeFile(molsys::SimulationBox &) {}
        virtual void readStressTensor(
            molsys::Box &,
            physicalData::PhysicalData &
        )
        {
        }

        void readForceFile(
            molsys::SimulationBox &,
            physicalData::PhysicalData &
        );
        void readChargeFile(molsys::SimulationBox &);

        /*******************************
         * standard getter and setters *
         *******************************/

        [[nodiscard]] const std::string &getScriptPath() const;
        [[nodiscard]] std::string        getSingularity() const;
        [[nodiscard]] std::string        getStaticBuild() const;

        void setScriptPath(const std::string_view &scriptPath);
    };
}   // namespace QM

#endif   // _EXTERNAL_QM_RUNNER_HPP_
