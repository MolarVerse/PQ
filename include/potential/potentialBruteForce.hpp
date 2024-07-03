#ifndef _POTENTIAL_BRUTE_FORCE_HPP_

#define _POTENTIAL_BRUTE_FORCE_HPP_

#include "potential.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class PotentialBruteForce
     *
     * @brief brute force implementation of the potential
     *
     */
    class PotentialBruteForce : public Potential
    {
       public:
        ~PotentialBruteForce() override;

        void calculateForces(pq::SimBox &, pq::PhysicalData &, pq::CellList &)
            override;

        pq::SharedPotential clone() const override;
    };
}   // namespace potential

#endif   // _POTENTIAL_BRUTE_FORCE_HPP_