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

#ifndef _HYBRID_MD_ENGINE_HPP_

#define _HYBRID_MD_ENGINE_HPP_

#include "mdEngine.hpp"
#include "qmCapableEngine.hpp"

namespace engine
{
    /**
     * @brief HybridMDEngine
     *
     * @details This class is a pure virtual class that inherits from MDEngine
     * and QMCapableEngine and is used to implement the Hybrid MD engine
     * backbone that can run in general combinations of MM and QM engines.
     *
     */
    class HybridMDEngine : virtual public MDEngine, public QMCapableEngine
    {
       public:
        using MDEngine::MDEngine;

        ~HybridMDEngine() override = default;

        void calculateForces() override = 0;

       protected:
        void _combineInnerOuterForces();

        void _addCurrentForcesToInnerAndReset(
            std::vector<std::shared_ptr<molsys::Atom>>& atoms
        );
        void _addScaledCurrentForcesToInnerAndReset(
            std::vector<std::shared_ptr<molsys::Atom>>& atoms,
            double                                      globalSmF
        );

        void _addCurrentForcesToOuterAndReset(
            std::vector<std::shared_ptr<molsys::Atom>>& atoms
        );
        void _addScaledCurrentForcesToOuterAndReset(
            std::vector<std::shared_ptr<molsys::Atom>>& atoms,
            double                                      globalSmF
        );

        void _scaleSmoothingMoleculeForcesInner();
        void _scaleSmoothingMoleculeForcesOuter();

        [[nodiscard]]
        static std::unordered_set<size_t> _generateInactiveSmoothingMoleculeSet(
            size_t bitPattern,
            size_t totalMolecules
        );

        [[nodiscard]]
        double _calculateGlobalSmoothingFactor(
            const std::unordered_set<size_t>& inactiveForInnerCalcMolecules
        ) const;
    };

}   // namespace engine

#endif   // _HYBRID_MD_ENGINE_HPP_
