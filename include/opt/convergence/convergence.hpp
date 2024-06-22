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
    };
}   // namespace opt

#endif   // _CONVERGENCE_HPP_