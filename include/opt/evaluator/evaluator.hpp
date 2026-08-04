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

#include <memory>

#include "celllist.hpp"
#include "constraints.hpp"
#include "forceFieldClass.hpp"
#include "hessianBuilder.hpp"
#include "intraNonBonded.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"
#include "typeAliases.hpp"
#include "virial.hpp"

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
        std::shared_ptr<potential::Potential>           _potential;
        std::shared_ptr<simulationBox::SimulationBox>   _simulationBox;
        std::shared_ptr<constraints::Constraints>       _constraints;
        std::shared_ptr<simulationBox::CellList>        _cellList;
        std::shared_ptr<forceField::ForceField>         _forceField;
        std::shared_ptr<physicalData::PhysicalData>     _physicalData;
        std::shared_ptr<physicalData::PhysicalData>     _physicalDataOld;
        std::shared_ptr<virial::Virial>                 _virial;
        std::shared_ptr<intraNonBonded::IntraNonBonded> _intraNonBonded;

       public:
        Evaluator()          = default;
        virtual ~Evaluator() = default;

        virtual std::shared_ptr<Evaluator> clone() const = 0;
        virtual void                       evaluate()    = 0;

        [[nodiscard]] virtual bool          supportsAnalyticHessian() const;
        [[nodiscard]] virtual HessianMatrix calculateAnalyticHessian();

        /***************************
         * standard setter methods *
         ***************************/

        void setPotential(const pq::SharedPotential);
        void setCellList(const pq::SharedCellList);
        void setSimulationBox(const pq::SharedSimBox);
        void setConstraints(const pq::SharedConstraints);

        void setPhysicalData(const std::shared_ptr<physicalData::PhysicalData>);
        void setPhysicalDataOld(
            const std::shared_ptr<physicalData::PhysicalData>
        );

        void setForceField(const std::shared_ptr<forceField::ForceField>);
        void setVirial(const std::shared_ptr<virial::Virial>);
        void setIntraNonBonded(const pq::SharedIntraNonBond);
    };

}   // namespace opt

#endif   // _EVALUATOR_HPP_
