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

#ifndef ENGINE__ENGINE_OUTPUT_HPP
#define ENGINE__ENGINE_OUTPUT_HPP

#include <cstddef>
#include <memory>
#include <mstd/types.hpp>
#include <optional>
#include <vector>

#include "boxOutput.hpp"
#include "energyOutput.hpp"
#include "infoOutput.hpp"
#include "logOutput.hpp"
#include "momentumOutput.hpp"
#include "optOutput.hpp"
#include "referencesOutput.hpp"
#include "ringPolymerEnergyOutput.hpp"
#include "ringPolymerRestartFileOutput.hpp"
#include "ringPolymerTrajectoryOutput.hpp"
#include "rstFileOutput.hpp"
#include "serialWorker.hpp"
#include "stdoutOutput.hpp"
#include "stressOutput.hpp"
#include "timingsOutput.hpp"
#include "trajectoryOutput.hpp"
#include "virialOutput.hpp"

namespace configurator
{
    class HybridConfigurator;   // forward declaration
}   // namespace configurator

namespace out
{
    struct Frame;   // forward declaration
}   // namespace out

namespace thermostat
{
    class Thermostat;   // forward declaration
}   // namespace thermostat

class OutputFileSettings;   // forward declaration

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
        mstd::ConstRef<OutputFileSettings> _outputFileSettings;

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

        references::ReferencesOutput _referencesOutput;

        // created on the first frame if async output is on and fits in
        // memory; null otherwise. Declared last so it is destroyed first: its
        // destructor waits for queued frames, which still use the writers
        // above.
        std::unique_ptr<utilities::SerialWorker> _frameWriter;
        bool                                     _frameWriterChecked = false;

        static constexpr std::size_t _frameCapacity = 2;

       public:
        explicit EngineOutput(
            const OutputFileSettings &outputFileSettings,
            const ReferenceSettings  &referenceSettings
        );

        void writeEnergyFile(
            size_t step,
            const physicalData::PhysicalData &,
            const Settings &
        );
        void writeInstantEnergyFile(
            size_t step,
            const physicalData::PhysicalData &,
            const Settings &
        );

        void writeXyzFile(const out::Frame &, size_t);
        void writeHybridCenterXyzFile(
            const configurator::HybridConfigurator &,
            size_t
        );
        void writeVelFile(const out::Frame &, size_t);
        void writeForceFile(const out::Frame &, size_t);
        void writeChargeFile(const out::Frame &, size_t);
        void writeRstFile(const out::Frame &, size_t);
        void writeFrame(
            const molsys::SimulationBox &,
            const thermostat::Thermostat &,
            size_t
        );
        void waitForFrames();

        [[nodiscard]] static bool asyncOutputFits(
            std::size_t                nAtoms,
            std::optional<std::size_t> availableMemory
        );
        void writeOptRstFile(molsys::SimulationBox &, size_t);

        void writeMomentumFile(size_t step, const physicalData::PhysicalData &);
        void writeVirialFile(size_t, const physicalData::PhysicalData &);
        void writeStressFile(size_t, const physicalData::PhysicalData &);
        void writeBoxFile(size_t, const molsys::Box &);
        void writeOptFile(size_t, const opt::Optimizer &);

        void writeRingPolymerRstFile(std::vector<molsys::SimulationBox> &);
        void writeRingPolymerXyzFile(
            std::vector<molsys::SimulationBox> &,
            size_t
        );
        void writeRingPolymerVelFile(
            std::vector<molsys::SimulationBox> &,
            size_t
        );
        void writeRingPolymerForceFile(
            std::vector<molsys::SimulationBox> &,
            size_t
        );
        void writeRingPolymerChargeFile(
            std::vector<molsys::SimulationBox> &,
            size_t
        );
        void writeRingPolymerEnergyFile(
            size_t,
            const std::vector<physicalData::PhysicalData> &
        );

        void writeTimingsFile();
        void writeReferencesFile();

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

        [[nodiscard]]
        out::RingPolymerRestartFileOutput &getRingPolymerRstFileOutput();
        [[nodiscard]]
        out::RingPolymerTrajectoryOutput &getRingPolymerXyzOutput();
        [[nodiscard]]
        out::RingPolymerTrajectoryOutput &getRingPolymerVelOutput();
        [[nodiscard]]
        out::RingPolymerTrajectoryOutput &getRingPolymerForceOutput();
        [[nodiscard]]
        out::RingPolymerTrajectoryOutput &getRingPolymerChargeOutput();
        [[nodiscard]]
        out::RingPolymerEnergyOutput &getRingPolymerEnergyOutput();

        [[nodiscard]] out::TimingsOutput &getTimingsOutput();

       private:
        void _startFrameWriter(std::size_t nAtoms);
        void _writeFrame(const out::Frame &, size_t step, bool metadata);
    };

}   // namespace engine

#endif   // ENGINE__ENGINE_OUTPUT_HPP
