#ifndef _VELOCITY_VERLET_HPP_

#define _VELOCITY_VERLET_HPP_

#include "integrator.hpp"
#include "typeAliases.hpp"

namespace integrator
{
    /**
     * @class VelocityVerlet inherits Integrator
     *
     * @brief VelocityVerlet is a class for velocity verlet integrator
     *
     */
    class VelocityVerlet : public Integrator
    {
       public:
        explicit VelocityVerlet();

        void firstStep(pq::SimBox &) override;
        void secondStep(pq::SimBox &) override;
    };

}   // namespace integrator

#endif   // _VELOCITY_VERLET_HPP_