#ifndef _ENGINE_OUTPUT_HPP_

#define _ENGINE_OUTPUT_HPP_

#include "energyOutput.hpp"                   // for EnergyOutput
#include "infoOutput.hpp"                     // for InfoOutput
#include "logOutput.hpp"                      // for LogOutput
#include "ringPolymerRestartFileOutput.hpp"   // for RingPolymerRestartFileOutput
#include "ringPolymerTrajectoryOutput.hpp"    // for RingPolymerTrajectoryOutput
#include "rstFileOutput.hpp"                  // for RstFileOutput
#include "stdoutOutput.hpp"                   // for StdoutOutput
#include "trajectoryOutput.hpp"               // for TrajectoryOutput

#include <cstddef>   // for size_t
#include <memory>    // for make_unique, unique_ptr

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
    /**
     * @class EngineOutput
     *
     * @brief contains unique pointers to all of the output classes
     *
     */
    class EngineOutput
    {
      private:
        std::unique_ptr<output::EnergyOutput>     _energyOutput  = std::make_unique<output::EnergyOutput>("default.en");
        std::unique_ptr<output::TrajectoryOutput> _xyzOutput     = std::make_unique<output::TrajectoryOutput>("default.xyz");
        std::unique_ptr<output::TrajectoryOutput> _velOutput     = std::make_unique<output::TrajectoryOutput>("default.vel");
        std::unique_ptr<output::TrajectoryOutput> _forceOutput   = std::make_unique<output::TrajectoryOutput>("default.force");
        std::unique_ptr<output::TrajectoryOutput> _chargeOutput  = std::make_unique<output::TrajectoryOutput>("default.chg");
        std::unique_ptr<output::LogOutput>        _logOutput     = std::make_unique<output::LogOutput>("default.log");
        std::unique_ptr<output::StdoutOutput>     _stdoutOutput  = std::make_unique<output::StdoutOutput>("stdout");
        std::unique_ptr<output::RstFileOutput>    _rstFileOutput = std::make_unique<output::RstFileOutput>("default.rst");
        std::unique_ptr<output::InfoOutput>       _infoOutput    = std::make_unique<output::InfoOutput>("default.info");

        std::unique_ptr<output::RingPolymerRestartFileOutput> _ringPolymerRstFileOutput =
            std::make_unique<output::RingPolymerRestartFileOutput>("default.rpmd.rst");
        std::unique_ptr<output::RingPolymerTrajectoryOutput> _ringPolymerXyzOutput =
            std::make_unique<output::RingPolymerTrajectoryOutput>("default.rpmd.xyz");
        std::unique_ptr<output::RingPolymerTrajectoryOutput> _ringPolymerVelOutput =
            std::make_unique<output::RingPolymerTrajectoryOutput>("default.rpmd.vel");
        std::unique_ptr<output::RingPolymerTrajectoryOutput> _ringPolymerForceOutput =
            std::make_unique<output::RingPolymerTrajectoryOutput>("default.rpmd.force");
        std::unique_ptr<output::RingPolymerTrajectoryOutput> _ringPolymerChargeOutput =
            std::make_unique<output::RingPolymerTrajectoryOutput>("default.rpmd.chg");

      public:
        void writeEnergyFile(const size_t step, const double loopTime, const physicalData::PhysicalData &);
        void writeXyzFile(simulationBox::SimulationBox &);
        void writeVelFile(simulationBox::SimulationBox &);
        void writeForceFile(simulationBox::SimulationBox &);
        void writeChargeFile(simulationBox::SimulationBox &);
        void writeInfoFile(const double simulationTime, const double loopTime, const physicalData::PhysicalData &);
        void writeRstFile(simulationBox::SimulationBox &, const size_t);

        void writeRingPolymerRstFile(std::vector<simulationBox::SimulationBox> &, const size_t);
        void writeRingPolymerXyzFile(std::vector<simulationBox::SimulationBox> &);
        void writeRingPolymerVelFile(std::vector<simulationBox::SimulationBox> &);
        void writeRingPolymerForceFile(std::vector<simulationBox::SimulationBox> &);
        void writeRingPolymerChargeFile(std::vector<simulationBox::SimulationBox> &);

        output::EnergyOutput                 &getEnergyOutput() { return *_energyOutput; }
        output::TrajectoryOutput             &getXyzOutput() { return *_xyzOutput; }
        output::TrajectoryOutput             &getVelOutput() { return *_velOutput; }
        output::TrajectoryOutput             &getForceOutput() { return *_forceOutput; }
        output::TrajectoryOutput             &getChargeOutput() { return *_chargeOutput; }
        output::LogOutput                    &getLogOutput() { return *_logOutput; }
        output::StdoutOutput                 &getStdoutOutput() { return *_stdoutOutput; }
        output::RstFileOutput                &getRstFileOutput() { return *_rstFileOutput; }
        output::InfoOutput                   &getInfoOutput() { return *_infoOutput; }
        output::RingPolymerRestartFileOutput &getRingPolymerRstFileOutput() { return *_ringPolymerRstFileOutput; }
        output::RingPolymerTrajectoryOutput  &getRingPolymerXyzOutput() { return *_ringPolymerXyzOutput; }
        output::RingPolymerTrajectoryOutput  &getRingPolymerVelOutput() { return *_ringPolymerVelOutput; }
        output::RingPolymerTrajectoryOutput  &getRingPolymerForceOutput() { return *_ringPolymerForceOutput; }
        output::RingPolymerTrajectoryOutput  &getRingPolymerChargeOutput() { return *_ringPolymerChargeOutput; }
    };

}   // namespace engine

#endif   // _ENGINE_OUTPUT_HPP_