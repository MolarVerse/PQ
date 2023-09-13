#ifndef _DFTBPLUS_RUNNER_HPP_

#define _DFTBPLUS_RUNNER_HPP_

#include "qmRunner.hpp"   // for QMRunner

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace QM
{
    /**
     * @class DFTBPlusRunner
     *
     * @brief class for running DFTB+ inheriting from QMRunner
     *
     */
    class DFTBPlusRunner : public QMRunner
    {
      private:
        bool _isFirstExecution = true;

      public:
        void writeCoordsFile(simulationBox::SimulationBox &) override;
        void execute() override;
        void readForceFile(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;
    };
}   // namespace QM

#endif   // _DFTBPLUS_RUNNER_HPP_