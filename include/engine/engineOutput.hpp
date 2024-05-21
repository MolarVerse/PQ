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

#ifndef _ENGINE_OUTPUT_HPP_

#define _ENGINE_OUTPUT_HPP_

#include <cstddef>   // for size_t
#include <memory>    // for make_unique, unique_ptr
#include <vector>    // for vector

#include "boxOutput.hpp"                      // for BoxFileOutput
#include "energyOutput.hpp"                   // for EnergyOutput
#include "infoOutput.hpp"                     // for InfoOutput
#include "logOutput.hpp"                      // for LogOutput
#include "momentumOutput.hpp"                 // for MomentumOutput
#include "ringPolymerEnergyOutput.hpp"        // for RingPolymerEnergyOutput
#include "ringPolymerRestartFileOutput.hpp"   // for RingPolymerRestartFileOutput
#include "ringPolymerTrajectoryOutput.hpp"    // for RingPolymerTrajectoryOutput
#include "rstFileOutput.hpp"                  // for RstFileOutput
#include "stdoutOutput.hpp"                   // for StdoutOutput
#include "stressOutput.hpp"                   // for StressOutput
#include "timer.hpp"                          // for Timer
#include "trajectoryOutput.hpp"               // for TrajectoryOutput
#include "virialOutput.hpp"                   // for VirialOutput

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace engine
{
    using namespace output;

    using RPMDRestartFileOutput = RingPolymerRestartFileOutput;
    using RPMDTrajectoryOutput  = RingPolymerTrajectoryOutput;
    using RPMDEnergyOutput      = RingPolymerEnergyOutput;

    using SimulationBox = simulationBox::SimulationBox;

    /**
     * @class EngineOutput
     *
     * @brief contains unique pointers to all of the output classes
     *
     */
    class EngineOutput : public timings::Timer
    {
       private:
        std::unique_ptr<EnergyOutput> _energyOutput;
        std::unique_ptr<EnergyOutput> _instantEnergyOutput;
        std::unique_ptr<InfoOutput>   _infoOutput;

        std::unique_ptr<TrajectoryOutput> _xyzOutput;
        std::unique_ptr<TrajectoryOutput> _velOutput;
        std::unique_ptr<TrajectoryOutput> _forceOutput;
        std::unique_ptr<TrajectoryOutput> _chargeOutput;
        std::unique_ptr<RstFileOutput>    _rstFileOutput;

        std::unique_ptr<LogOutput>    _logOutput;
        std::unique_ptr<StdoutOutput> _stdoutOutput;

        std::unique_ptr<MomentumOutput> _momentumOutput;
        std::unique_ptr<VirialOutput>   _virialOutput;
        std::unique_ptr<StressOutput>   _stressOutput;
        std::unique_ptr<BoxFileOutput>  _boxFileOutput;

        std::unique_ptr<RPMDRestartFileOutput> _ringPolymerRstFileOutput;
        std::unique_ptr<RPMDTrajectoryOutput>  _ringPolymerXyzOutput;
        std::unique_ptr<RPMDTrajectoryOutput>  _ringPolymerVelOutput;
        std::unique_ptr<RPMDTrajectoryOutput>  _ringPolymerForceOutput;
        std::unique_ptr<RPMDTrajectoryOutput>  _ringPolymerChargeOutput;
        std::unique_ptr<RPMDEnergyOutput>      _ringPolymerEnergyOutput;

       public:
        EngineOutput();

        void writeEnergyFile(const size_t step, const physicalData::PhysicalData &);
        void writeInstantEnergyFile(const size_t step, const physicalData::PhysicalData &);
        void writeMomentumFile(const size_t step, const physicalData::PhysicalData &);
        void writeXyzFile(simulationBox::SimulationBox &);
        void writeVelFile(simulationBox::SimulationBox &);
        void writeForceFile(simulationBox::SimulationBox &);
        void writeChargeFile(simulationBox::SimulationBox &);
        void writeInfoFile(const double simulationTime, const physicalData::PhysicalData &);
        void writeRstFile(simulationBox::SimulationBox &, const size_t);

        void writeVirialFile(const size_t, const physicalData::PhysicalData &);
        void writeStressFile(const size_t, const physicalData::PhysicalData &);
        void writeBoxFile(const size_t, const simulationBox::Box &);

        void writeRingPolymerRstFile(
            std::vector<SimulationBox> &,
            const size_t
        );
        void writeRingPolymerXyzFile(std::vector<SimulationBox> &);
        void writeRingPolymerVelFile(std::vector<SimulationBox> &);
        void writeRingPolymerForceFile(std::vector<SimulationBox> &);
        void writeRingPolymerChargeFile(std::vector<SimulationBox> &);
        void writeRingPolymerEnergyFile(const size_t, const std::vector<physicalData::PhysicalData> &);

        EnergyOutput &getEnergyOutput() { return *_energyOutput; }
        EnergyOutput &getInstantEnergyOutput() { return *_instantEnergyOutput; }
        MomentumOutput   &getMomentumOutput() { return *_momentumOutput; }
        TrajectoryOutput &getXyzOutput() { return *_xyzOutput; }
        TrajectoryOutput &getVelOutput() { return *_velOutput; }
        TrajectoryOutput &getForceOutput() { return *_forceOutput; }
        TrajectoryOutput &getChargeOutput() { return *_chargeOutput; }
        LogOutput        &getLogOutput() { return *_logOutput; }
        StdoutOutput     &getStdoutOutput() { return *_stdoutOutput; }
        RstFileOutput    &getRstFileOutput() { return *_rstFileOutput; }
        InfoOutput       &getInfoOutput() { return *_infoOutput; }

        VirialOutput  &getVirialOutput() { return *_virialOutput; }
        StressOutput  &getStressOutput() { return *_stressOutput; }
        BoxFileOutput &getBoxFileOutput() { return *_boxFileOutput; }

        RingPolymerRestartFileOutput &getRingPolymerRstFileOutput();
        RingPolymerTrajectoryOutput  &getRingPolymerXyzOutput();
        RingPolymerTrajectoryOutput  &getRingPolymerVelOutput();
        RingPolymerTrajectoryOutput  &getRingPolymerForceOutput();
        RingPolymerTrajectoryOutput  &getRingPolymerChargeOutput();
        RingPolymerEnergyOutput      &getRingPolymerEnergyOutput();
    };

}   // namespace engine

#endif   // _ENGINE_OUTPUT_HPP_