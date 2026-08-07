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

#include "boxOutput.hpp"
#include "energyOutput.hpp"
#include "infoOutput.hpp"
#include "logOutput.hpp"
#include "momentumOutput.hpp"
#include "optOutput.hpp"
#include "ringPolymerEnergyOutput.hpp"
#include "ringPolymerRestartFileOutput.hpp"
#include "ringPolymerTrajectoryOutput.hpp"
#include "rstFileOutput.hpp"
#include "stdoutOutput.hpp"
#include "stressOutput.hpp"
#include "timer.hpp"   // for Timer
#include "timingsOutput.hpp"
#include "trajectoryOutput.hpp"
#include "typeAliases.hpp"
#include "virialOutput.hpp"

namespace engine
{
    /**
     * @class EngineOutput
     *
     * @brief contains unique pointers to all of the output classes
     *
     */
    class EngineOutput : public timings::Timer
    {
       private:
        std::unique_ptr<pq::EnergyOutput> _energyOutput;
        std::unique_ptr<pq::EnergyOutput> _instantEnergyOutput;
        std::unique_ptr<pq::InfoOutput>   _infoOutput;

        std::unique_ptr<pq::TrajectoryOutput> _xyzOutput;
        std::unique_ptr<pq::TrajectoryOutput> _velOutput;
        std::unique_ptr<pq::TrajectoryOutput> _forceOutput;
        std::unique_ptr<pq::TrajectoryOutput> _chargeOutput;
        std::unique_ptr<pq::RstFileOutput>    _rstFileOutput;

        std::unique_ptr<pq::LogOutput>    _logOutput;
        std::unique_ptr<pq::StdoutOutput> _stdoutOutput;

        std::unique_ptr<pq::MomentumOutput> _momentumOutput;
        std::unique_ptr<pq::VirialOutput>   _virialOutput;
        std::unique_ptr<pq::StressOutput>   _stressOutput;
        std::unique_ptr<pq::BoxFileOutput>  _boxFileOutput;

        std::unique_ptr<pq::OptOutput> _optOutput;

        pq::UniqueRPMDRstFileOutput _rpmdRstFileOutput;
        pq::UniqueRPMDTrajOutput    _rpmdXyzOutput;
        pq::UniqueRPMDTrajOutput    _rpmdVelOutput;
        pq::UniqueRPMDTrajOutput    _rpmdForceOutput;
        pq::UniqueRPMDTrajOutput    _rpmdChargeOutput;
        pq::UniqueRPMDEnergyOutput  _rpmdEnergyOutput;

        std::unique_ptr<pq::TimingsOutput> _timingsOutput;

       public:
        EngineOutput();

        void writeEnergyFile(const size_t step, const pq::PhysicalData &);
        void writeInstantEnergyFile(const size_t step, const pq::PhysicalData &);
        void writeXyzFile(pq::SimBox &);
        void writeVelFile(pq::SimBox &);
        void writeForceFile(pq::SimBox &);
        void writeChargeFile(pq::SimBox &);
        void writeInfoFile(const double simulationTime, const pq::PhysicalData &);
        void writeRstFile(pq::SimBox &, const pq::Thermostat &, const size_t);
        void writeOptRstFile(pq::SimBox &, const size_t);

        void writeMomentumFile(const size_t step, const pq::PhysicalData &);
        void writeVirialFile(const size_t, const pq::PhysicalData &);
        void writeStressFile(const size_t, const pq::PhysicalData &);
        void writeBoxFile(const size_t, const pq::Box &);
        void writeOptFile(const size_t, const pq::Optimizer &);

        void writeRingPolymerRstFile(std::vector<pq::SimBox> &, const size_t);
        void writeRingPolymerXyzFile(std::vector<pq::SimBox> &);
        void writeRingPolymerVelFile(std::vector<pq::SimBox> &);
        void writeRingPolymerForceFile(std::vector<pq::SimBox> &);
        void writeRingPolymerChargeFile(std::vector<pq::SimBox> &);
        void writeRingPolymerEnergyFile(const size_t, const std::vector<pq::PhysicalData> &);

        void writeTimingsFile(timings::GlobalTimer &);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] pq::EnergyOutput     &getEnergyOutput();
        [[nodiscard]] pq::EnergyOutput     &getInstantEnergyOutput();
        [[nodiscard]] pq::TrajectoryOutput &getXyzOutput();
        [[nodiscard]] pq::TrajectoryOutput &getVelOutput();
        [[nodiscard]] pq::TrajectoryOutput &getForceOutput();
        [[nodiscard]] pq::TrajectoryOutput &getChargeOutput();
        [[nodiscard]] pq::RstFileOutput    &getRstFileOutput();
        [[nodiscard]] pq::InfoOutput       &getInfoOutput();

        [[nodiscard]] pq::LogOutput    &getLogOutput();
        [[nodiscard]] pq::StdoutOutput &getStdoutOutput();

        [[nodiscard]] pq::MomentumOutput &getMomentumOutput();
        [[nodiscard]] pq::VirialOutput   &getVirialOutput();
        [[nodiscard]] pq::StressOutput   &getStressOutput();
        [[nodiscard]] pq::BoxFileOutput  &getBoxFileOutput();

        [[nodiscard]] pq::OptOutput &getOptOutput();

        [[nodiscard]] pq::RPMDRstFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerXyzOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerVelOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerForceOutput();
        [[nodiscard]] pq::RPMDTrajOutput    &getRingPolymerChargeOutput();
        [[nodiscard]] pq::RPMDEnergyOutput  &getRingPolymerEnergyOutput();

        [[nodiscard]] pq::TimingsOutput &getTimingsOutput();
    };

}   // namespace engine

#endif   // _ENGINE_OUTPUT_HPP_