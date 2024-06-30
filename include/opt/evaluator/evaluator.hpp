/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _EVALUATOR_HPP_

#define _EVALUATOR_HPP_

#include <memory>   // for shared_ptr

#include "typeAliases.hpp"

namespace opt
{
    /**
     * @class Evaluator
     *
     * @brief Base class for all evaluators (e.g. MM, QM, ...)
     *        Evaluators are used to evaluate forces/hessians
     *
     */
    class Evaluator
    {
       protected:
        pq::SharedPotential    _potential;
        pq::SharedSimBox       _simulationBox;
        pq::SharedConstraints  _constraints;
        pq::SharedCellList     _cellList;
        pq::SharedForceField   _forceField;
        pq::SharedPhysicalData _physicalData;
        pq::SharedPhysicalData _physicalDataOld;
        pq::SharedVirial       _virial;
        pq::SharedIntraNonBond _intraNonBonded;

       public:
        Evaluator()          = default;
        virtual ~Evaluator() = default;

        virtual pq::SharedEvaluator clone() const = 0;
        virtual void                evaluate()    = 0;

        /***************************
         * standard setter methods *
         ***************************/

        void setPotential(const pq::SharedPotential);
        void setCellList(const pq::SharedCellList);
        void setSimulationBox(const pq::SharedSimBox);
        void setConstraints(const pq::SharedConstraints);
        void setPhysicalData(const pq::SharedPhysicalData);
        void setPhysicalDataOld(const pq::SharedPhysicalData);
        void setForceField(const pq::SharedForceField);
        void setVirial(const pq::SharedVirial);
        void setIntraNonBonded(const pq::SharedIntraNonBond);
    };

}   // namespace opt

#endif   // _EVALUATOR_HPP_