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

#ifndef _CONVERGENCE_HPP_

#define _CONVERGENCE_HPP_

#include "typeAliases.hpp"   // for pq::ConvStrategy

namespace opt
{
    /**
     * @brief Convergence class
     *
     * @details This class stores all information about the convergence of the
     * optimization
     */
    class Convergence
    {
       protected:
        bool _isRelEnergyConv   = true;
        bool _isAbsEnergyConv   = true;
        bool _isAbsMaxForceConv = true;
        bool _isAbsRMSForceConv = true;

        bool _enableEnergyConv   = true;
        bool _enableMaxForceConv = true;
        bool _enableRMSForceConv = true;

        double _relEnergy   = 0.0;
        double _absEnergy   = 0.0;
        double _absMaxForce = 0.0;
        double _absRMSForce = 0.0;

        double _relEnergyConvThreshold   = 0.0;
        double _absEnergyConvThreshold   = 0.0;
        double _absMaxForceConvThreshold = 0.0;
        double _absRMSForceConvThreshold = 0.0;

        pq::ConvStrategy _energyConvStrategy;

       public:
        Convergence() = default;
        Convergence(
            const bool,
            const bool,
            const bool,
            const double,
            const double,
            const double,
            const double,
            const pq::ConvStrategy
        );

        [[nodiscard]] bool checkConvergence() const;

        void calcEnergyConvergence(const double, const double);
        void calcForceConvergence(const double, const double);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] double getRelEnergy() const;
        [[nodiscard]] double getAbsEnergy() const;
        [[nodiscard]] double getAbsMaxForce() const;
        [[nodiscard]] double getAbsRMSForce() const;

        [[nodiscard]] pq::ConvStrategy getEnConvStrategy() const;

        [[nodiscard]] bool isEnergyConvEnabled() const;
        [[nodiscard]] bool isMaxForceConvEnabled() const;
        [[nodiscard]] bool isRMSForceConvEnabled() const;

        [[nodiscard]] bool isRelEnergyConv() const;
        [[nodiscard]] bool isAbsEnergyConv() const;
        [[nodiscard]] bool isAbsMaxForceConv() const;
        [[nodiscard]] bool isAbsRMSForceConv() const;

        [[nodiscard]] double getRelEnergyConvThreshold() const;
        [[nodiscard]] double getAbsEnergyConvThreshold() const;
        [[nodiscard]] double getAbsMaxForceConvThreshold() const;
        [[nodiscard]] double getAbsRMSForceConvThreshold() const;
    };
}   // namespace opt

#endif   // _CONVERGENCE_HPP_