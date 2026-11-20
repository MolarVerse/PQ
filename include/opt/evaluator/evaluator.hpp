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

#ifndef OPT__EVALUATOR__EVALUATOR_HPP
#define OPT__EVALUATOR__EVALUATOR_HPP

#include <memory>
#include <mstd/types.hpp>

#include "celllist.hpp"
#include "constraints.hpp"
#include "forceFieldClass.hpp"
#include "hessianBuilder.hpp"
#include "intraNonBonded.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "simulationBox.hpp"
#include "virial.hpp"

struct Settings;

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
       private:
        mstd::ConstRef<Settings>                        _settings;
        molsys::CellList*                               _cellList;
        std::shared_ptr<pot::Potential>                 _potential;
        std::shared_ptr<molsys::SimulationBox>          _simulationBox;
        std::shared_ptr<constraints::Constraints>       _constraints;
        std::shared_ptr<ff::ForceField>                 _forceField;
        std::shared_ptr<physicalData::PhysicalData>     _physicalData;
        std::shared_ptr<physicalData::PhysicalData>     _physicalDataOld;
        std::shared_ptr<intraNonBonded::IntraNonBonded> _intraNonBonded;

       public:
        explicit Evaluator(const Settings& settings);
        virtual ~Evaluator();

        [[nodiscard]] virtual std::shared_ptr<Evaluator> clone() const = 0;
        virtual void                                     evaluate()    = 0;

        [[nodiscard]] virtual bool          supportsAnalyticHessian() const;
        [[nodiscard]] virtual HessianMatrix calculateAnalyticHessian();

        /***************************
         * standard setter methods *
         ***************************/

        void setPotential(const std::shared_ptr<pot::Potential>&);
        void setCellList(molsys::CellList&);
        void setSimulationBox(const std::shared_ptr<molsys::SimulationBox>&);
        void setConstraints(const std::shared_ptr<constraints::Constraints>&);

        void setPhysicalData(
            const std::shared_ptr<physicalData::PhysicalData>&
        );
        void setPhysicalDataOld(
            const std::shared_ptr<physicalData::PhysicalData>&
        );

        void setForceField(const std::shared_ptr<ff::ForceField>&);
        void setIntraNonBonded(
            const std::shared_ptr<intraNonBonded::IntraNonBonded>&
        );

       protected:
        [[nodiscard]]
        physicalData::PhysicalData& _getPhysicalData();
        [[nodiscard]]
        const physicalData::PhysicalData& _getPhysicalData() const;

        [[nodiscard]]
        physicalData::PhysicalData& _getPhysicalDataOld();
        [[nodiscard]]
        const physicalData::PhysicalData& _getPhysicalDataOld() const;

        [[nodiscard]]
        molsys::SimulationBox& _getSimulationBox();
        [[nodiscard]]
        const molsys::SimulationBox& _getSimulationBox() const;

        [[nodiscard]]
        molsys::CellList& _getCellList();
        [[nodiscard]]
        const molsys::CellList& _getCellList() const;

        [[nodiscard]]
        pot::Potential& _getPotential();
        [[nodiscard]]
        const pot::Potential& _getPotential() const;

        [[nodiscard]]
        ff::ForceField& _getForceField();
        [[nodiscard]]
        const ff::ForceField& _getForceField() const;

        [[nodiscard]]
        constraints::Constraints& _getConstraints();
        [[nodiscard]]
        const constraints::Constraints& _getConstraints() const;

        [[nodiscard]]
        intraNonBonded::IntraNonBonded& _getIntraNonBonded();
        [[nodiscard]]
        const intraNonBonded::IntraNonBonded& _getIntraNonBonded() const;

        [[nodiscard]]
        const Settings& _getSettings() const;
    };

}   // namespace opt

#endif   // OPT__EVALUATOR__EVALUATOR_HPP
