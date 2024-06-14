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

namespace constraints
{
    class Constraints;   // forward declaration
}   // namespace constraints

namespace forceField
{
    class ForceField;   // forward declaration
}   // namespace forceField

namespace intraNonBonded
{
    class IntraNonBonded;   // forward declaration
}   // namespace intraNonBonded

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace potential
{
    class Potential;   // forward declaration
}   // namespace potential

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class CellList;        // forward declaration
}   // namespace simulationBox

namespace virial
{
    class Virial;   // forward declaration
}   // namespace virial

namespace opt
{
    using SharedCellList     = std::shared_ptr<simulationBox::CellList>;
    using SharedSimBox       = std::shared_ptr<simulationBox::SimulationBox>;
    using SharedForceField   = std::shared_ptr<forceField::ForceField>;
    using SharedPotential    = std::shared_ptr<potential::Potential>;
    using SharedPhysicalData = std::shared_ptr<physicalData::PhysicalData>;
    using SharedConstraints  = std::shared_ptr<constraints::Constraints>;
    using SharedIntraNonBond = std::shared_ptr<intraNonBonded::IntraNonBonded>;
    using SharedVirial       = std::shared_ptr<virial::Virial>;

    /**
     * @class Evaluator
     *
     * @brief Base class for all evaluators (e.g. MM, QM, ...)
     *        Evaluators are used to evaluate forces/hessians
     *
     */
    class Evaluator
    {
       private:
        std::shared_ptr<potential::Potential>           _potential;
        std::shared_ptr<simulationBox::SimulationBox>   _simulationBox;
        std::shared_ptr<constraints::Constraints>       _constraints;
        std::shared_ptr<simulationBox::CellList>        _cellList;
        std::shared_ptr<forceField::ForceField>         _forceField;
        std::shared_ptr<physicalData::PhysicalData>     _physicalData;
        std::shared_ptr<virial::Virial>                 _virial;
        std::shared_ptr<intraNonBonded::IntraNonBonded> _intraNonBonded;

       public:
        Evaluator()          = default;
        virtual ~Evaluator() = default;

        void setPotential(const SharedPotential);
        void setSimulationBox(const SharedSimBox);
        void setConstraints(const SharedConstraints);
        void setCellList(const SharedCellList);
        void setForceField(const SharedForceField);
        void setPhysicalData(const SharedPhysicalData);
        void setVirial(const SharedVirial);
        void setIntraNonBonded(const SharedIntraNonBond);
    };

}   // namespace opt

#endif   // _EVALUATOR_HPP_