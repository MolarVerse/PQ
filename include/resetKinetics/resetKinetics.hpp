#ifndef _RESET_KINETICS_HPP_

#define _RESET_KINETICS_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <cstddef>   // for size_t

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace resetKinetics
{
    /**
     * @class ResetKinetics
     *
     * @brief base class for the reset of the kinetics - represents also class for no reset
     *
     */
    class ResetKinetics
    {
      protected:
        size_t _nStepsTemperatureReset;
        size_t _frequencyTemperatureReset;
        size_t _nStepsMomentumReset;
        size_t _frequencyMomentumReset;
        size_t _nStepsAngularReset;
        size_t _frequencyAngularReset;

        double               _temperature = 0.0;
        linearAlgebra::Vec3D _momentum;
        linearAlgebra::Vec3D _angularMomentum;

      public:
        ResetKinetics() = default;
        ResetKinetics(const size_t nStepsTemperatureReset,
                      const size_t frequencyTemperatureReset,
                      const size_t nStepsMomentumReset,
                      const size_t frequencyMomentumReset,
                      const size_t nStepsAngularReset,
                      const size_t frequencyAngularReset)
            : _nStepsTemperatureReset(nStepsTemperatureReset), _frequencyTemperatureReset(frequencyTemperatureReset),
              _nStepsMomentumReset(nStepsMomentumReset), _frequencyMomentumReset(frequencyMomentumReset),
              _nStepsAngularReset(nStepsAngularReset), _frequencyAngularReset(frequencyAngularReset){};

        void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &);
        void resetTemperature(simulationBox::SimulationBox &);
        void resetMomentum(simulationBox::SimulationBox &);
        void resetAngularMomentum(simulationBox::SimulationBox &);

        /********************
         * standard setters *
         *******************/

        void setTemperature(const double temperature) { _temperature = temperature; }
        void setMomentum(const linearAlgebra::Vec3D &momentum) { _momentum = momentum; }
        void setAngularMomentum(const linearAlgebra::Vec3D &angularMomentum) { _angularMomentum = angularMomentum; }

        /********************
         * standard getters *
         *******************/

        [[nodiscard]] size_t getNStepsTemperatureReset() const { return _nStepsTemperatureReset; }
        [[nodiscard]] size_t getFrequencyTemperatureReset() const { return _frequencyTemperatureReset; }
        [[nodiscard]] size_t getNStepsMomentumReset() const { return _nStepsMomentumReset; }
        [[nodiscard]] size_t getFrequencyMomentumReset() const { return _frequencyMomentumReset; }
    };

}   // namespace resetKinetics

#endif   // _RESET_KINETICS_HPP_