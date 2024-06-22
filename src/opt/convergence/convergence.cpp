#include "convergence.hpp"

using namespace opt;

/**
 * @brief Construct a new Convergence object
 *
 * @param absEnergyConv
 * @param relEnergyConv
 * @param absMaxForceConv
 * @param absEnergyConvThreshold
 * @param relEnergyConvThreshold
 * @param absMaxForceConvThreshold
 * @param absRMSForceConvThreshold
 * @param energyConvStrategy
 */
Convergence::Convergence(
    const bool                   _enableEnergyConv,
    const bool                   _enableMaxForceConv,
    const bool                   _enableRMSForceConv,
    const double                 relEnergyConvThreshold,
    const double                 absEnergyConvThreshold,
    const double                 absMaxForceConvThreshold,
    const double                 absRMSForceConvThreshold,
    const settings::ConvStrategy energyConvStrategy
)
    : _enableEnergyConv(_enableEnergyConv),
      _enableMaxForceConv(_enableMaxForceConv),
      _enableRMSForceConv(_enableRMSForceConv),
      _relEnergyConv(relEnergyConvThreshold),
      _absEnergyConv(absEnergyConvThreshold),
      _absMaxForceConv(absMaxForceConvThreshold),
      _absRMSForceConv(absRMSForceConvThreshold),
      _energyConvStrategy(energyConvStrategy)
{
}

/**
 * @brief check if the optimizer has converged
 *
 * @return true/false if the optimizer has converged
 */
bool Convergence::checkConvergence() const
{
    auto isEnergyConverged = true;

    if (_energyConvStrategy == settings::ConvStrategy::RIGOROUS)
        isEnergyConverged = _isAbsEnergyConv && _isRelEnergyConv;

    else if (_energyConvStrategy == settings::ConvStrategy::LOOSE)
        isEnergyConverged = _isAbsEnergyConv || _isRelEnergyConv;

    else if (_energyConvStrategy == settings::ConvStrategy::ABSOLUTE)
        isEnergyConverged = _isAbsEnergyConv;

    else if (_energyConvStrategy == settings::ConvStrategy::RELATIVE)
        isEnergyConverged = _isRelEnergyConv;

    return isEnergyConverged && _isAbsMaxForceConv && _isAbsRMSForceConv;
}

/**
 * @brief calculate the energy convergence
 *
 * @param energyOld
 * @param energyNew
 */
void Convergence::calcEnergyConvergence(
    const double energyOld,
    const double energyNew
)
{
    _absEnergy = std::abs(energyNew - energyOld);
    _relEnergy = _absEnergy / std::abs(energyOld);

    if (_enableEnergyConv)
    {
        _isAbsEnergyConv = _absEnergy < _absEnergyConv;
        _isRelEnergyConv = _relEnergy < _relEnergyConv;
    }
}

/**
 * @brief calculate the force convergence
 *
 * @param forceOld
 * @param forceNew
 */
void Convergence::calcForceConvergence(
    const double maxForce,
    const double rmsForce
)
{
    _absMaxForce = std::abs(maxForce);
    _absRMSForce = std::abs(rmsForce);

    if (_enableMaxForceConv)
        _isAbsMaxForceConv = _absMaxForce < _absMaxForceConv;

    if (_enableRMSForceConv)
        _isAbsRMSForceConv = _absRMSForce < _absRMSForceConv;
}
