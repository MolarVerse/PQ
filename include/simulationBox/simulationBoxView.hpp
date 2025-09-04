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

#ifndef _SIMULATION_BOX_VIEW_HPP_

#define _SIMULATION_BOX_VIEW_HPP_

#include "views.hpp"

namespace simulationBox
{
    /**
     * @class SimulationBoxView
     *
     * @brief
     *
     *  CRTP based class for all kind of views on atoms, molecules etc.
     *
     */
    template <typename Derived>
    class SimulationBoxView
    {
       private:
        auto& getAtoms() const;
        auto& getAtoms();

        auto& getMolecules() const;
        auto& getMolecules();

       public:
        auto getQMAtoms();
        auto getQMAtoms() const;

        auto getMMAtoms();
        auto getMMAtoms() const;

        auto getQMAtomicNumbers() const;

        auto getMoleculesInsideZone(const HybridZone) const;
        auto getMoleculesInsideZone(const HybridZone);

        auto getMoleculesOutsideZone(const HybridZone) const;
        auto getMoleculesOutsideZone(const HybridZone);

        auto getInactiveMolecules();
        auto getInactiveMolecules() const;
    };

    /**
     * @brief Get the Atoms vector reference
     *
     * @return const auto& a reference to the atoms vector
     */
    template <typename Derived>
    auto& SimulationBoxView<Derived>::getAtoms() const
    {
        return static_cast<const Derived&>(*this).getAtoms();
    }

    /**
     * @brief Get the Atoms vector reference
     *
     * @return auto& a reference to the atoms vector
     */
    template <typename Derived>
    auto& SimulationBoxView<Derived>::getAtoms()
    {
        return static_cast<Derived&>(*this).getAtoms();
    }

    /**
     * @brief Get the Molecules vector reference
     *
     * @return const auto& a reference to the molecules vector
     */
    template <typename Derived>
    auto& SimulationBoxView<Derived>::getMolecules() const
    {
        return static_cast<const Derived&>(*this).getMolecules();
    }

    /**
     * @brief Get the Molecules vector reference
     *
     * @return auto& a reference to the molecules vector
     */
    template <typename Derived>
    auto& SimulationBoxView<Derived>::getMolecules()
    {
        return static_cast<Derived&>(*this).getMolecules();
    }

    /**
     * @brief get all QM atoms using range-based filtering
     *
     * @return a view/iterator of QM atoms filtered from all atoms
     *
     * @details This function returns a range-based view that filters atoms
     *          from _atoms based on whether they are designated as QM
     * atoms.
     */
    template <typename Derived>
    auto SimulationBoxView<Derived>::getQMAtoms()
    {
        return getAtoms() |
               pqviews::filter([](auto& atom) { return atom->isQMAtom(); });
    }

    /**
     * @brief get all QM atoms using range-based filtering
     *
     * @return a view/iterator of QM atoms filtered from all atoms
     *
     * @details This function returns a range-based view that filters atoms
     *          from _atoms based on whether they are designated as QM
     * atoms.
     */
    template <typename Derived>
    auto SimulationBoxView<Derived>::getQMAtoms() const
    {
        return getAtoms() | pqviews::filter([](const auto& atom)
                                            { return atom->isQMAtom(); });
    }

    /**
     * @brief get all MM atoms using range-based filtering
     *
     * @return a view/iterator of MM atoms filtered from all atoms
     *
     * @details This function returns a range-based view that filters atoms
     *          from _atoms based on whether they are designated as MM
     * atoms.
     */
    template <typename Derived>
    auto SimulationBoxView<Derived>::getMMAtoms()
    {
        return getAtoms() |
               pqviews::filter([](auto& atom) { return atom->isMMAtom(); });
    }

    /**
     * @brief get all MM atoms using range-based filtering
     *
     * @return a view/iterator of MM atoms filtered from all atoms
     *
     * @details This function returns a range-based view that filters atoms
     *          from _atoms based on whether they are designated as MM
     * atoms.
     */
    template <typename Derived>
    auto SimulationBoxView<Derived>::getMMAtoms() const
    {
        return getAtoms() | pqviews::filter([](const auto& atom)
                                            { return atom->isMMAtom(); });
    }

    /**
     * @brief get all QM atomic numbers using range-based filtering
     *
     * @return a view/iterator of QM atomic numbers
     */
    template <typename Derived>
    auto SimulationBoxView<Derived>::getQMAtomicNumbers() const
    {
        return getQMAtoms() |
               pqviews::transform([](const auto& atom)
                                  { return atom->getAtomicNumber(); });
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
    auto SimulationBoxView<Derived>::getMoleculesInsideZone(
        const HybridZone zone
    )
    {
        return getMolecules() |
               pqviews::filter([zone](auto& mol)
                               { return mol.getHybridZone() == zone; });
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
    auto SimulationBoxView<Derived>::getMoleculesInsideZone(
        const HybridZone zone
    ) const
    {
        return getMolecules() |
               pqviews::filter([zone](const auto& mol)
                               { return mol.getHybridZone() == zone; });
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
    auto SimulationBoxView<Derived>::getMoleculesOutsideZone(
        const HybridZone zone
    )
    {
        return getMolecules() |
               pqviews::filter([zone](auto& mol)
                               { return mol.getHybridZone() != zone; });
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
    auto SimulationBoxView<Derived>::getMoleculesOutsideZone(
        const HybridZone zone
    ) const
    {
        return getMolecules() |
               pqviews::filter([zone](const auto& mol)
                               { return mol.getHybridZone() != zone; });
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
    auto SimulationBoxView<Derived>::getInactiveMolecules()
    {
        return getMolecules() |
               pqviews::filter([](auto& mol) { return !mol.isActive(); });
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
    auto SimulationBoxView<Derived>::getInactiveMolecules() const
    {
        return getMolecules() |
               pqviews::filter([](const auto& mol) { return !mol.isActive(); });
    }

}   // namespace simulationBox

#endif   // _SIMULATION_BOX_VIEW_HPP_