#ifndef _RESET_KINETICS_HPP_

#define _RESET_KINETICS_HPP_

#include "physicalData.hpp"
#include "simulationBox.hpp"

#include <cstddef>

namespace resetKinetics
{
    class ResetKinetics;
    class ResetMomentum;
    class ResetTemperature;
}   // namespace resetKinetics

/**
 * @class ResetKinetics
 *
 * @brief base class for the reset of the kinetics
 *
 */
class resetKinetics::ResetKinetics
{
  protected:
    size_t _nStepsTemperatureReset;
    size_t _frequencyTemperatureReset;
    size_t _nStepsMomentumReset;
    size_t _frequencyMomentumReset;

    double _targetTemperature;

  public:
    ResetKinetics() = default;
    ResetKinetics(const size_t nStepsTemperatureReset,
                  const size_t frequencyTemperatureReset,
                  const size_t nStepsMomentumReset,
                  const size_t frequencyMomentumReset,
                  const double targetTemperature)
        : _nStepsTemperatureReset(nStepsTemperatureReset), _frequencyTemperatureReset(frequencyTemperatureReset),
          _nStepsMomentumReset(nStepsMomentumReset), _frequencyMomentumReset(frequencyMomentumReset),
          _targetTemperature(targetTemperature){};

    virtual ~ResetKinetics() = default;

    virtual void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &) const;
    void         resetTemperature(const physicalData::PhysicalData &, simulationBox::SimulationBox &) const;
    void         resetMomentum(physicalData::PhysicalData &, simulationBox::SimulationBox &) const;

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
class resetKinetics::ResetMomentum : public resetKinetics::ResetKinetics
{
  public:
    using resetKinetics::ResetKinetics::ResetKinetics;

    void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &) const override;
};

/**
 * @class ResetTemperature inherits from ResetKinetics
 *
 * @brief reset the temperature and the momentum of the system
 *
 */
class resetKinetics::ResetTemperature : public resetKinetics::ResetKinetics
{
  public:
    using resetKinetics::ResetKinetics::ResetKinetics;

    void reset(const size_t step, physicalData::PhysicalData &, simulationBox::SimulationBox &) const override;
};

#endif   // _RESET_KINETICS_HPP_