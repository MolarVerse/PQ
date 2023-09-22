#ifndef _RESET_KINETICS_HPP_

#define _RESET_KINETICS_HPP_

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

      public:
        ResetKinetics() = default;
        ResetKinetics(const size_t nStepsAngularReset, const size_t frequencyAngularReset)
            : _nStepsAngularReset(nStepsAngularReset), _frequencyAngularReset(frequencyAngularReset){};
        ResetKinetics(const size_t nStepsTemperatureReset,
                      const size_t frequencyTemperatureReset,
                      const size_t nStepsMomentumReset,
                      const size_t frequencyMomentumReset,
                      const size_t nStepsAngularReset,
                      const size_t frequencyAngularReset)
            : _nStepsTemperatureReset(nStepsTemperatureReset), _frequencyTemperatureReset(frequencyTemperatureReset),
              _nStepsMomentumReset(nStepsMomentumReset), _frequencyMomentumReset(frequencyMomentumReset),
              _nStepsAngularReset(nStepsAngularReset), _frequencyAngularReset(frequencyAngularReset){};

        virtual ~ResetKinetics() = default;

        virtual void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &) const;
        void         resetTemperature(physicalData::PhysicalData &, simulationBox::SimulationBox &) const;
        void         resetMomentum(physicalData::PhysicalData &, simulationBox::SimulationBox &) const;
        void         resetAngularMomentum(physicalData::PhysicalData &, simulationBox::SimulationBox &) const;

        /********************
         * standard getters *
         *******************/

        [[nodiscard]] size_t getNStepsTemperatureReset() const { return _nStepsTemperatureReset; }
        [[nodiscard]] size_t getFrequencyTemperatureReset() const { return _frequencyTemperatureReset; }
        [[nodiscard]] size_t getNStepsMomentumReset() const { return _nStepsMomentumReset; }
        [[nodiscard]] size_t getFrequencyMomentumReset() const { return _frequencyMomentumReset; }
    };

    /**
     * @class ResetMomentum inherits from ResetKinetics
     *
     * @brief reset the momentum of the system
     *
     */
    class ResetMomentum : public ResetKinetics
    {
      public:
        using ResetKinetics::ResetKinetics;

        void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &) const override;
    };

    /**
     * @class ResetTemperature inherits from ResetKinetics
     *
     * @brief reset the temperature and the momentum of the system
     *
     */
    class ResetTemperature : public ResetKinetics
    {
      public:
        using ResetKinetics::ResetKinetics;

        void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &) const override;
    };

}   // namespace resetKinetics

#endif   // _RESET_KINETICS_HPP_