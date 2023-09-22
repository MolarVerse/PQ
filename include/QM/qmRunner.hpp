#ifndef _QM_RUNNER_HPP_

#define _QM_RUNNER_HPP_

#define SCRIPT_PATH_ _SCRIPT_PATH_

#include <string>

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
     * @class QMRunner
     *
     * @brief base class for different qm engines
     *
     */
    class QMRunner
    {
      protected:
        const std::string _scriptPath = SCRIPT_PATH_;

      public:
        virtual ~QMRunner() = default;

        void         run(simulationBox::SimulationBox &, physicalData::PhysicalData &);
        virtual void writeCoordsFile(simulationBox::SimulationBox &) = 0;
        virtual void execute()                                       = 0;
        virtual void readForceFile(simulationBox::SimulationBox &, physicalData::PhysicalData &);
    };
}   // namespace QM

#endif   // _QM_RUNNER_HPP_