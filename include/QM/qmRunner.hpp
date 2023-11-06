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

#ifndef _QM_RUNNER_HPP_

#define _QM_RUNNER_HPP_

#define SCRIPT_PATH_ _SCRIPT_PATH_
#define SINGULARITY_ _SINGULARITY_

#include <string>

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Box;             // forward declaration

}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace QM
{
    /**
     * @class QMRunner
     *
     * @brief base class for different qm engines
     *
     */
    class QMRunner
    {
      protected:
        std::string       _scriptPath  = SCRIPT_PATH_;
        const std::string _singularity = SINGULARITY_;

      public:
        virtual ~QMRunner() = default;

        void         run(simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void writeCoordsFile(simulationBox::SimulationBox &) = 0;
        virtual void execute()                                       = 0;
        virtual void readForceFile(simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void readStressTensor(simulationBox::Box &, physicalData::PhysicalData &){};

        /*******************************
         * standard getter and setters *
         *******************************/

        [[nodiscard]] const std::string &getScriptPath() const { return _scriptPath; }
        [[nodiscard]] const std::string &getSingularity() const { return _singularity; }

        void setScriptPath(const std::string_view &scriptPath) { _scriptPath = scriptPath; }
    };
}   // namespace QM

#endif   // _QM_RUNNER_HPP_