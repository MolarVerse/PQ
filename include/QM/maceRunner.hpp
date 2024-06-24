#ifndef _MACE_RUNNER_HPP_

#define _MACE_RUNNER_HPP_

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "internalQMRunner.hpp"   // for InternalQMRunner

namespace QM
{
    /**
     * @brief MaceRunner inherits from InternalQMRunner
     *
     */
    class __attribute__((visibility("default"))) MaceRunner
        : public InternalQMRunner
    {
       private:
        double                    _energy;
        pybind11::object          _calculator;
        pybind11::object          _atoms_module;
        pybind11::array_t<double> _forces;
        pybind11::array_t<double> _stress_tensor;

        std::unique_ptr<pybind11::scoped_interpreter> _guard;

       public:
        explicit MaceRunner(const std::string &model);
        ~MaceRunner() override = default;

        void execute() override;
        void prepareAtoms(pq::SimBox &) override;
        void collectData(pq::SimBox &, pq::PhysicalData &) override;
    };
}   // namespace QM

#endif   // _MACE_RUNNER_HPP_