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

#ifndef _CELL_VIEW_HPP_

#define _CELL_VIEW_HPP_

#include "molecule.hpp"
#include "views.hpp"

namespace simulationBox
{
    /**
     * @class CellView
     *
     * @brief
     *
     *  CRTP based class for all kind of views on molecules etc.
     *
     */
    template <typename Derived>
    class CellView
    {
       private:
        auto& getMolecules() const;
        auto& getMolecules();

       public:
        auto getMMMolecules();
        auto getMMMolecules() const;

        auto getMoleculesInsideZone(const HybridZone) const;
        auto getMoleculesInsideZone(const HybridZone);

        auto getMoleculesOutsideZone(const HybridZone) const;
        auto getMoleculesOutsideZone(const HybridZone);

        auto getInactiveMolecules();
        auto getInactiveMolecules() const;
    };

    /**
     * @brief Get the Molecules vector reference
     *
     * @return const auto& a reference to the molecules vector
     */
    template <typename Derived>
    auto& CellView<Derived>::getMolecules() const
    {
        return static_cast<const Derived&>(*this).getMolecules();
    }

    /**
     * @brief Get the Molecules vector reference
     *
     * @return auto& a reference to the molecules vector
     */
    template <typename Derived>
    auto& CellView<Derived>::getMolecules()
    {
        return static_cast<Derived&>(*this).getMolecules();
    }

    /**
     * @brief get all MM molecules using range-based filtering
     *
     * @return a view/iterator of MM molecules filtered from all molecules
     *
     * @details This function returns a range-based view that filters molecules
     *          from _molecules based on whether they are designated as
     * molecular mechanics (MM) molecules. The classification depends on the job
     * type:
     *          - MM-only simulations: all molecules are MM molecules
     *          - QM-only simulations: no molecules are MM molecules
     *          - Hybrid QM/MM simulations: active molecules are MM molecules
     */
    template <typename Derived>
    auto CellView<Derived>::getMMMolecules()
    {
        return getMolecules() |
               pqviews::filter([](auto& mol) { return mol->isMMMolecule(); });
    }

    /**
     * @brief get all MM molecules using range-based filtering
     *
     * @return a view/iterator of MM molecules filtered from all molecules
     *
     * @details This function returns a range-based view that filters molecules
     *          from _molecules based on whether they are designated as
     * molecular mechanics (MM) molecules. The classification depends on the job
     * type:
     *          - MM-only simulations: all molecules are MM molecules
     *          - QM-only simulations: no molecules are MM molecules
     *          - Hybrid QM/MM simulations: active molecules are MM molecules
     */
    template <typename Derived>
    auto CellView<Derived>::getMMMolecules() const
    {
        return getMolecules() |
               pqviews::filter([](const auto& mol)
                               { return mol->isMMMolecule(); });
    }

    /**
     * @brief get all molecules in the specified hybrid zone using range-based
     * filtering
     *
     * @return a view/iterator of molecules in the specified zone filtered from
     * all molecules
     *
     * @details This function returns a range-based view that filters molecules
     * from _molecules based on whether they are in the specified HybridZone
     */
    template <typename Derived>
    auto CellView<Derived>::getMoleculesInsideZone(const HybridZone zone)
    {
        return getMolecules() |
               pqviews::filter([zone](auto& mol)
                               { return mol->getHybridZone() == zone; });
    }

    /**
     * @brief get all molecules in the specified hybrid zone using range-based
     * filtering
     *
     * @return a view/iterator of molecules in the specified zone filtered from
     * all molecules
     *
     * @details This function returns a range-based view that filters molecules
     * from _molecules based on whether they are in the specified HybridZone
     */
    template <typename Derived>
    auto CellView<Derived>::getMoleculesInsideZone(const HybridZone zone) const
    {
        return getMolecules() |
               pqviews::filter([zone](const auto& mol)
                               { return mol->getHybridZone() == zone; });
    }

    /**
     * @brief get all molecules outside the specified hybrid zone using
     * range-based filtering
     *
     * @return a view/iterator of molecules outside the specified zone filtered
     * from all molecules
     *
     * @details This function returns a range-based view that filters molecules
     * from _molecules based on whether they are outside the specified
     * HybridZone
     */
    template <typename Derived>
    auto CellView<Derived>::getMoleculesOutsideZone(const HybridZone zone)
    {
        return getMolecules() |
               pqviews::filter([zone](auto& mol)
                               { return mol->getHybridZone() != zone; });
    }

    /**
     * @brief get all molecules outside the specified hybrid zone using range-based
     * filtering
     *
     * @return a view/iterator of molecules outside the specified zone filtered from
     * all molecules
     *
     * @details This function returns a range-based view that filters molecules
     * from _molecules based on whether they are outside the specified HybridZone
     */
    template <typename Derived>
    auto CellView<Derived>::getMoleculesOutsideZone(const HybridZone zone) const
    {
        return getMolecules() |
               pqviews::filter([zone](const auto& mol)
                               { return mol->getHybridZone() != zone; });
    }

    /**
     * @brief get all inactive molecules using range-based filtering
     *
     * @return a view/iterator of inactive molecules filtered from all
     * molecules
     *
     * @details This function returns a range-based view that filters molecules
     * from _molecules based on whether they are inactive (i.e., not active).
     */
    template <typename Derived>
    auto CellView<Derived>::getInactiveMolecules()
    {
        return getMolecules() |
               pqviews::filter([](auto& mol) { return !mol->isActive(); });
    }

    /**
     * @brief get all inactive molecules using range-based filtering
     *
     * @return a view/iterator of inactive molecules filtered from all
     * molecules
     *
     * @details This function returns a range-based view that filters molecules
     * from _molecules based on whether they are inactive (i.e., not active).
     */
    template <typename Derived>
    auto CellView<Derived>::getInactiveMolecules() const
    {
        return getMolecules() | pqviews::filter([](const auto& mol)
                                                { return !mol->isActive(); });
    }

}   // namespace simulationBox

#endif   // _CELL_VIEW_HPP_