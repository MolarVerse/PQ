#ifndef _THERMOSTAT_HPP_

#define _THERMOSTAT_HPP_

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

/**
 * @namespace thermostat
 */
namespace thermostat
{
    /**
     * @class Thermostat
     *
     * @brief Thermostat is a base class for all thermostats
     *
     * @details it provides a dummy function applyThermostat() which does only calculate the temperature
     *
     */
    class Thermostat
    {
      protected:
        double _temperature       = 0.0;
        double _targetTemperature = 0.0;
        double _timestep          = 0.0;

      public:
        Thermostat() = default;
        explicit Thermostat(const double targetTemperature) : _targetTemperature(targetTemperature) {}
        virtual ~Thermostat() = default;

        virtual void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);

        void                 setTimestep(const double timestep) { _timestep = timestep; }
        [[nodiscard]] double getTimestep() const { return _timestep; }
    };

    /**
     * @class BerendsenThermostat
     *
     * @brief BerendsenThermostat is a class for the Berendsen thermostat
     *
     * @link https://doi.org/10.1063/1.448118
     *
     */
    class BerendsenThermostat : public Thermostat
    {
      private:
        double _tau;

      public:
        BerendsenThermostat() = default;
        explicit BerendsenThermostat(const double targetTemp, const double tau) : Thermostat(targetTemp), _tau(tau) {}

        void applyThermostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;

        [[nodiscard]] double getTau() const { return _tau; }
        void                 setTau(const double tau) { _tau = tau; }
    };

}   // namespace thermostat

#endif   // _THERMOSTAT_HPP_