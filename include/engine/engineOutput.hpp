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
#include "timingsOutput.hpp"
#include "trajectoryOutput.hpp"
#include "virialOutput.hpp"

namespace configurator
{
    class HybridConfigurator;   // forward declaration
}   // namespace configurator

namespace engine
{
    /**
     * @class EngineOutput
     *
     * @brief contains unique pointers to all of the output classes
     *
     */
    class EngineOutput
    {
       private:
        std::unique_ptr<out::EnergyOutput> _energyOutput;
        std::unique_ptr<out::EnergyOutput> _instantEnergyOutput;
        std::unique_ptr<out::InfoOutput>   _infoOutput;

        std::unique_ptr<out::TrajectoryOutput> _xyzOutput;
        std::unique_ptr<out::TrajectoryOutput> _xyzHybridCenterOutput;
        std::unique_ptr<out::TrajectoryOutput> _velOutput;
        std::unique_ptr<out::TrajectoryOutput> _forceOutput;
        std::unique_ptr<out::TrajectoryOutput> _chargeOutput;
        std::unique_ptr<out::RstFileOutput>    _rstFileOutput;

        std::unique_ptr<out::LogOutput>    _logOutput;
        std::unique_ptr<out::StdoutOutput> _stdoutOutput;

        std::unique_ptr<out::MomentumOutput> _momentumOutput;
        std::unique_ptr<out::VirialOutput>   _virialOutput;
        std::unique_ptr<out::StressOutput>   _stressOutput;
        std::unique_ptr<out::BoxFileOutput>  _boxFileOutput;

        std::unique_ptr<out::OptOutput> _optOutput;

        std::unique_ptr<out::RingPolymerRestartFileOutput> _rpmdRstFileOutput;
        std::unique_ptr<out::RingPolymerTrajectoryOutput>  _rpmdXyzOutput;
        std::unique_ptr<out::RingPolymerTrajectoryOutput>  _rpmdVelOutput;
        std::unique_ptr<out::RingPolymerTrajectoryOutput>  _rpmdForceOutput;
        std::unique_ptr<out::RingPolymerTrajectoryOutput>  _rpmdChargeOutput;
        std::unique_ptr<out::RingPolymerEnergyOutput>      _rpmdEnergyOutput;

        std::unique_ptr<out::TimingsOutput> _timingsOutput;

       public:
        EngineOutput();

        void writeEnergyFile(
            const size_t step,
            const physicalData::PhysicalData &
        );
        void writeInstantEnergyFile(
            const size_t step,
            const physicalData::PhysicalData &
        );

        void writeXyzFile(molsys::SimulationBox &, const size_t);
        void writeHybridCenterXyzFile(
            const configurator::HybridConfigurator &,
            const size_t
        );
        void writeVelFile(molsys::SimulationBox &, const size_t);
        void writeForceFile(molsys::SimulationBox &, const size_t);
        void writeChargeFile(molsys::SimulationBox &, const size_t);
        void writeInfoFile(
            const double simulationTime,
            const physicalData::PhysicalData &
        );
        void writeRstFile(
            molsys::SimulationBox &,
            const thermostat::Thermostat &,
            const size_t
        );
        void writeOptRstFile(molsys::SimulationBox &, const size_t);

        void writeMomentumFile(
            const size_t step,
            const physicalData::PhysicalData &
        );
        void writeVirialFile(const size_t, const physicalData::PhysicalData &);
        void writeStressFile(const size_t, const physicalData::PhysicalData &);
        void writeBoxFile(const size_t, const molsys::Box &);
        void writeOptFile(const size_t, const opt::Optimizer &);

        void writeRingPolymerRstFile(std::vector<molsys::SimulationBox> &);
        void writeRingPolymerXyzFile(
            std::vector<molsys::SimulationBox> &,
            const size_t
        );
        void writeRingPolymerVelFile(
            std::vector<molsys::SimulationBox> &,
            const size_t
        );
        void writeRingPolymerForceFile(
            std::vector<molsys::SimulationBox> &,
            const size_t
        );
        void writeRingPolymerChargeFile(
            std::vector<molsys::SimulationBox> &,
            const size_t
        );
        void writeRingPolymerEnergyFile(
            const size_t,
            const std::vector<physicalData::PhysicalData> &
        );

        void writeTimingsFile();

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] out::EnergyOutput     &getEnergyOutput();
        [[nodiscard]] out::EnergyOutput     &getInstantEnergyOutput();
        [[nodiscard]] out::TrajectoryOutput &getXyzOutput();
        [[nodiscard]] out::TrajectoryOutput &getXyzHybridCenterOutput();
        [[nodiscard]] out::TrajectoryOutput &getVelOutput();
        [[nodiscard]] out::TrajectoryOutput &getForceOutput();
        [[nodiscard]] out::TrajectoryOutput &getChargeOutput();
        [[nodiscard]] out::RstFileOutput    &getRstFileOutput();
        [[nodiscard]] out::InfoOutput       &getInfoOutput();

        [[nodiscard]] out::LogOutput    &getLogOutput();
        [[nodiscard]] out::StdoutOutput &getStdoutOutput();

        [[nodiscard]] out::MomentumOutput &getMomentumOutput();
        [[nodiscard]] out::VirialOutput   &getVirialOutput();
        [[nodiscard]] out::StressOutput   &getStressOutput();
        [[nodiscard]] out::BoxFileOutput  &getBoxFileOutput();

        [[nodiscard]] out::OptOutput &getOptOutput();

        // clang-format off
        [[nodiscard]] out::RingPolymerRestartFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]] out::RingPolymerTrajectoryOutput &getRingPolymerXyzOutput();
        [[nodiscard]] out::RingPolymerTrajectoryOutput &getRingPolymerVelOutput();
        [[nodiscard]] out::RingPolymerTrajectoryOutput &getRingPolymerForceOutput();
        [[nodiscard]] out::RingPolymerTrajectoryOutput &getRingPolymerChargeOutput();
        [[nodiscard]] out::RingPolymerEnergyOutput &getRingPolymerEnergyOutput();
        // clang-format on

        [[nodiscard]] out::TimingsOutput &getTimingsOutput();
    };

}   // namespace engine

#endif   // _ENGINE_OUTPUT_HPP_
