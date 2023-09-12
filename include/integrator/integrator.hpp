#ifndef _INTEGRATOR_HPP_

#define _INTEGRATOR_HPP_

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
    class Atom;            // forward declaration
}   // namespace simulationBox

namespace integrator
{
    /**
     * @class Integrator
     *
     * @brief Integrator is a base class for all integrators
     *
     */
    class Integrator
    {
      protected:
        std::string _integratorType;

      public:
        Integrator() = default;
        explicit Integrator(const std::string_view integratorType) : _integratorType(integratorType){};
        virtual ~Integrator() = default;

        virtual void firstStep(simulationBox::SimulationBox &)  = 0;
        virtual void secondStep(simulationBox::SimulationBox &) = 0;

        void integrateVelocities(simulationBox::Atom *) const;
        void integratePositions(simulationBox::Atom *, const simulationBox::SimulationBox &) const;

        /********************************
         * standard getters and setters *
         ********************************/

        [[nodiscard]] std::string_view getIntegratorType() const { return _integratorType; }
    };

    /**
     * @class VelocityVerlet inherits Integrator
     *
     * @brief VelocityVerlet is a class for velocity verlet integrator
     *
     */
    class VelocityVerlet : public Integrator
    {
      public:
        explicit VelocityVerlet() : Integrator("VelocityVerlet"){};

        void firstStep(simulationBox::SimulationBox &) override;
        void secondStep(simulationBox::SimulationBox &) override;
    };

}   // namespace integrator

#endif   // _INTEGRATOR_HPP_