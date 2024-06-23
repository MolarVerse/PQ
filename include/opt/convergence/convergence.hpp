#ifndef _CONVERGENCE_HPP_

#define _CONVERGENCE_HPP_

#include "convergenceSettings.hpp"

namespace opt
{
    /**
     * @brief Convergence class
     *
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

        double _relEnergyConv;
        double _absEnergyConv;
        double _absMaxForceConv;
        double _absRMSForceConv;

        settings::ConvStrategy _energyConvStrategy;

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
            const settings::ConvStrategy
        );

        bool checkConvergence() const;

        void calcEnergyConvergence(const double, const double);
        void calcForceConvergence(const double, const double);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] double getRelEnergy() const;
        [[nodiscard]] double getAbsEnergy() const;
        [[nodiscard]] double getAbsMaxForce() const;
        [[nodiscard]] double getAbsRMSForce() const;

        [[nodiscard]] settings::ConvStrategy getEnergyConvStrategy() const;

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