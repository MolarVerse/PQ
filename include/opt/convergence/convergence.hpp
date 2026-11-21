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

#ifndef OPT__CONVERGENCE__CONVERGENCE_HPP
#define OPT__CONVERGENCE__CONVERGENCE_HPP

#include "enums/convergence.hpp"
#include "unit_types.hpp"

namespace opt
{
    /**
     * @brief Convergence class
     *
     */
    class Convergence
    {
       private:
        bool _isRelEnergyConv   = true;
        bool _isAbsEnergyConv   = true;
        bool _isAbsMaxForceConv = true;
        bool _isAbsRMSForceConv = true;

        bool _enableEnergyConv   = true;
        bool _enableMaxForceConv = true;
        bool _enableRMSForceConv = true;

        Quantity     _relEnergy   = 0.0;
        EnergyPerMol _absEnergy   = EnergyPerMol::zero();
        Force        _absMaxForce = Force::zero();
        Force        _absRMSForce = Force::zero();

        Quantity     _relEnergyConvThreshold   = 0.0;
        EnergyPerMol _absEnergyConvThreshold   = EnergyPerMol::zero();
        Force        _absMaxForceConvThreshold = Force::zero();
        Force        _absRMSForceConvThreshold = Force::zero();

        ConvStrategy _energyConvStrategy;

       public:
        Convergence() = default;
        Convergence(
            bool,
            bool,
            bool,
            const Quantity&,
            const EnergyPerMol&,
            const Force&,
            const Force&,
            ConvStrategy
        );

        [[nodiscard]] bool checkConvergence() const;

        void calcEnergyConvergence(const EnergyPerMol&, const EnergyPerMol&);
        void calcForceConvergence(const Force&, const Force&);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] const Quantity&     getRelEnergy() const;
        [[nodiscard]] const EnergyPerMol& getAbsEnergy() const;
        [[nodiscard]] const Force&        getAbsMaxForce() const;
        [[nodiscard]] const Force&        getAbsRMSForce() const;

        [[nodiscard]] ConvStrategy getEnConvStrategy() const;

        [[nodiscard]] bool isEnergyConvEnabled() const;
        [[nodiscard]] bool isMaxForceConvEnabled() const;
        [[nodiscard]] bool isRMSForceConvEnabled() const;

        [[nodiscard]] bool isRelEnergyConv() const;
        [[nodiscard]] bool isAbsEnergyConv() const;
        [[nodiscard]] bool isAbsMaxForceConv() const;
        [[nodiscard]] bool isAbsRMSForceConv() const;

        [[nodiscard]] const Quantity&     getRelEnergyConvThreshold() const;
        [[nodiscard]] const EnergyPerMol& getAbsEnergyConvThreshold() const;
        [[nodiscard]] const Force&        getAbsMaxForceConvThreshold() const;
        [[nodiscard]] const Force&        getAbsRMSForceConvThreshold() const;
    };
}   // namespace opt

#endif   // OPT__CONVERGENCE__CONVERGENCE_HPP
