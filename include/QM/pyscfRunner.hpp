#ifndef _PYSCF_RUNNER_HPP_

#define _PYSCF_RUNNER_HPP_

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
     * @class PySCFRunner
     *
     * @brief class for running PySCF inheriting from QMRunner
     *
     */
    class PySCFRunner : public QMRunner
    {
      private:
        bool _isFirstExecution = true;

      public:
        void writeCoordsFile(simulationBox::SimulationBox &) override;
        void execute() override;
    };

}   // namespace QM

#endif   // _PYSCF_RUNNER_HPP_