#ifndef _QM_RUNNER_HPP_

#define _QM_RUNNER_HPP_

namespace simulationBox
{
    class SimulationBox;   // forward declaration
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
      public:
        virtual ~QMRunner() = default;

        virtual void writeCoordsFile(simulationBox::SimulationBox &) = 0;
    };
}   // namespace QM

#endif   // _QM_RUNNER_HPP_