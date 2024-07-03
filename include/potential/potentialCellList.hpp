#ifndef _POTENTIAL_CELL_LIST_HPP_

#define _POTENTIAL_CELL_LIST_HPP_

#include "potential.hpp"
#include "typeAliases.hpp"

namespace potential
{
    /**
     * @class PotentialCellList
     *
     * @brief cell list implementation of the potential
     *
     */
    class PotentialCellList : public Potential
    {
       public:
        ~PotentialCellList() override;

        void calculateForces(pq::SimBox &, pq::PhysicalData &, pq::CellList &)
            override;

        pq::SharedPotential clone() const override;
    };

}   // namespace potential

#endif   // _POTENTIAL_CELL_LIST_HPP_