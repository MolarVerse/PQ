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

#ifndef _QM_MM_MD_ENGINE_HPP_

#define _QM_MM_MD_ENGINE_HPP_

#include <unordered_set>   // for unordered_set

#include "hybridMDEngine.hpp"

namespace engine
{
    /**
     * @brief class QMMMMDEngine
     *
     * @details This class is a class that inherits from HybridMDEngine
     * and is used to implement the QM/MM MD engine.
     *
     */
    class QMMMMDEngine : public HybridMDEngine
    {
       public:
        QMMMMDEngine()  = default;
        ~QMMMMDEngine() = default;

        void calculateForces() override;

       private:
        void applyExactSmoothing();
        void applyHotspotSmoothing();

        void setNumberOfQMAtoms();
        void moltypeCheck();

        [[nodiscard]] std::unordered_set<size_t> generateInactiveMoleculeSet(
            size_t bitPattern,
            size_t totalMolecules
        );
        [[nodiscard]] double calculateGlobalSmoothingFactor(
            const std::unordered_set<size_t>& inactiveForInnerCalcMolecules
        );
    };

}   // namespace engine

#endif   // _QM_MM_MD_ENGINE_HPP_