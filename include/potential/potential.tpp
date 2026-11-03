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

#ifndef POTENTIAL__POTENTIAL_TPP
#define POTENTIAL__POTENTIAL_TPP

#include <cmath>
#include <cstdlib>

#include "atom.hpp"
#include "hybridSettings.hpp"
#include "potential.hpp"

namespace pot
{

    /**
     * @brief make shared pointer of the Coulomb potential
     *
     * @tparam T
     * @param potential
     */
    template <typename T>
    void Potential::makeCoulombPotential(T potential)
    {
        _coulombPotential = std::make_shared<T>(potential);
    }

    /**
     * @brief make shared pointer of the non-Coulomb potential
     *
     * @tparam T
     * @param nonCoulombPotential
     */
    template <typename T>
    void Potential::makeNonCoulombPotential(const T &nonCoulombPotential)
    {
        _nonCoulombPot = std::make_shared<T>(nonCoulombPotential);
    }

    /**
     * @brief Generic template function for getting partial charge from an atom
     *
     * @details This base template should never be called directly. It serves as
     * a fallback that will abort the program if called with an unsupported
     * charge tag type. Only the specialized versions (QMChargeTag and
     * MMChargeTag) should be used.
     *
     * @tparam T Charge tag type (should be QMChargeTag or MMChargeTag)
     * @param atom Reference to the atom from which to extract the charge
     * @return double The partial charge (this implementation aborts)
     *
     * @throws std::abort() Always aborts as this should never be called
     */
    template <typename T>
    double Potential::getPartialCharge(molsys::Atom & /*atom*/) const
    {
        std::abort();
    }

    /**
     * @brief Template specialization for getting QM partial charge from an atom
     *
     * @details This specialization handles quantum mechanical charge
     * extraction. It first attempts to use the QM charge if available
     * (calculated from QM methods), and falls back to the standard partial
     * charge if no QM charge is present.
     *
     * @param atom Reference to the atom from which to extract the QM charge
     * @return double The QM charge if available, otherwise the standard partial
     * charge
     */
    template <>
    inline double Potential::getPartialCharge<QMChargeTag>(
        molsys::Atom &atom
    ) const
    {
        const auto useQMCharges = settings::HybridSettings::getUseQMCharges();

        if (atom.getQMCharge() && useQMCharges)
            return atom.getQMCharge().value();

        return atom.getPartialCharge();
    }

    /**
     * @brief Template specialization for getting MM partial charge from an atom
     *
     * @details This specialization always returns the standard partial charge
     * used in classical force fields.
     *
     * @param atom Reference to the atom from which to extract the MM charge
     * @return double The molecular mechanics partial charge
     */
    template <>
    inline double Potential::getPartialCharge<MMChargeTag>(
        molsys::Atom &atom
    ) const
    {
        return atom.getPartialCharge();
    }

}   // namespace pot

#endif   // POTENTIAL__POTENTIAL_TPP
