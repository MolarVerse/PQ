#ifndef _CONSTANTS_HPP_

#define _CONSTANTS_HPP_

#include <cmath>

namespace constants
{
    /**
     * @brief conversion factors for degrees
     *
     */
    static constexpr double _DEG_TO_RAD_ = M_PI / 180.0;
    static constexpr double _RAD_TO_DEG_ = 180.0 / M_PI;

    /**
     * @brief Conversion factors for mass units
     */
    static constexpr double _AMU_TO_KG_       = 1.660539040e-27;
    static constexpr double _KG_TO_AMU_       = 1.0 / _AMU_TO_KG_;
    static constexpr double _AVOGADRO_NUMBER_ = _KG_TO_AMU_ / 1000;

    /**
     * @brief Conversion factors for length units
     */
    static constexpr double _ANGSTROM_TO_METER_ = 1.0e-10;
    static constexpr double _METER_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_METER_;

    static constexpr double _BOHR_RADIUS_TO_METER_ = 5.2917721067e-11;
    static constexpr double _METER_TO_BOHR_RADIUS_ = 1.0 / _BOHR_RADIUS_TO_METER_;

    static constexpr double _ANGSTROM_TO_BOHR_RADIUS_ = _ANGSTROM_TO_METER_ / _BOHR_RADIUS_TO_METER_;
    static constexpr double _BOHR_RADIUS_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_BOHR_RADIUS_;

    /**
     * @brief Conversion factors for volume units
     */
    static constexpr double _ANGSTROM_CUBIC_TO_METER_CUBIC_ = _ANGSTROM_TO_METER_ * _ANGSTROM_TO_METER_ * _ANGSTROM_TO_METER_;
    static constexpr double _METER_CUBIC_TO_ANGSTROM_CUBIC_ = 1.0 / _ANGSTROM_CUBIC_TO_METER_CUBIC_;
    static constexpr double _ANGSTROM_CUBIC_TO_LITER_       = _ANGSTROM_CUBIC_TO_METER_CUBIC_ * 1.0e3;
    static constexpr double _LITER_TO_ANGSTROM_CUBIC_       = 1.0 / _ANGSTROM_CUBIC_TO_LITER_;

    /**
     * @brief Conversion factors for density units
     */
    static constexpr double _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_       = _KG_TO_AMU_ / _LITER_TO_ANGSTROM_CUBIC_;
    static constexpr double _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_ = 1.0 / _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_;

    /**
     * @brief Conversion factors for energy units
     */
    static constexpr double _KCAL_TO_JOULE_         = 4184.0;
    static constexpr double _JOULE_TO_KCAL_         = 1.0 / _KCAL_TO_JOULE_;
    static constexpr double _JOULE_TO_KCAL_PER_MOL_ = _JOULE_TO_KCAL_ * _AVOGADRO_NUMBER_;
    static constexpr double _KCAL_PER_MOL_TO_JOULE_ = 1.0 / _JOULE_TO_KCAL_PER_MOL_;

    /**
     * @brief Conversion factors for charge related data
     */
    static constexpr double _ELECTRON_CHARGE_         = 1.6021766208e-19;   // in Coulomb
    static constexpr double _ELECTRON_CHARGE_SQUARED_ = _ELECTRON_CHARGE_ * _ELECTRON_CHARGE_;
    static constexpr double _ELECTRON_MASS_           = 9.10938356e-31;    // in kg
    static constexpr double _PERMITTIVITY_VACUUM_     = 8.854187817e-12;   // in F/m

    /**
     * @brief Conversion factors for time units
     */
    static constexpr double _S_TO_FS_  = 1.0e15;
    static constexpr double _FS_TO_S_  = 1.0 / _S_TO_FS_;
    static constexpr double _PS_TO_FS_ = 1.0e3;
    static constexpr double _FS_TO_PS_ = 1.0 / _PS_TO_FS_;

    /**
     * @brief Conversion factors to SI units
     */
    static constexpr double _FORCE_UNIT_TO_SI_    = _KCAL_TO_JOULE_ / _AVOGADRO_NUMBER_ / _ANGSTROM_TO_METER_;
    static constexpr double _MASS_UNIT_TO_SI_     = _AMU_TO_KG_;
    static constexpr double _TIME_UNIT_TO_SI_     = _FS_TO_S_;
    static constexpr double _VELOCITY_UNIT_TO_SI_ = _ANGSTROM_TO_METER_;
    static constexpr double _SI_TO_VELOCITY_UNIT_ = 1.0 / _VELOCITY_UNIT_TO_SI_;
    static constexpr double _ENERGY_UNIT_TO_SI_   = _KCAL_TO_JOULE_ / _AVOGADRO_NUMBER_;
    static constexpr double _VOLUME_UNIT_TO_SI_   = _ANGSTROM_CUBIC_TO_METER_CUBIC_;

    /**
     * @brief Conversion factors for velocity verlet integrator
     *
     */
    static constexpr double _V_VERLET_VELOCITY_FACTOR_ =
        0.5 * (_FORCE_UNIT_TO_SI_ / _MASS_UNIT_TO_SI_) * _TIME_UNIT_TO_SI_ * _SI_TO_VELOCITY_UNIT_;

    /**
     * @brief Conversion factors for temperature calculation
     */
    static constexpr double _BOLTZMANN_CONSTANT_ = 1.38064852e-23;   // in J/K
    static constexpr double _TEMPERATURE_FACTOR_ =
        _VELOCITY_UNIT_TO_SI_ * _VELOCITY_UNIT_TO_SI_ * _MASS_UNIT_TO_SI_ / _BOLTZMANN_CONSTANT_;

    /**
     * @brief Conversion factors kinetic energy
     */
    static constexpr double _KINETIC_ENERGY_FACTOR_ =
        0.5 * _MASS_UNIT_TO_SI_ * _VELOCITY_UNIT_TO_SI_ * _VELOCITY_UNIT_TO_SI_ * _JOULE_TO_KCAL_PER_MOL_;

    /**
     * @brief Conversion factors for pressure calculation
     */
    static constexpr double _PASCAL_TO_BAR_   = 1.0e-5;
    static constexpr double _BAR_TO_PASCAL_   = 1.0 / _PASCAL_TO_BAR_;
    static constexpr double _PRESSURE_FACTOR_ = _ENERGY_UNIT_TO_SI_ / _VOLUME_UNIT_TO_SI_ * _PASCAL_TO_BAR_;

    /**
     * @brief Conversion factors for coulomb preFactor
     */
    static constexpr double _COULOMB_PREFACTOR_ =
        1 / (4 * M_PI * _PERMITTIVITY_VACUUM_) * _ELECTRON_CHARGE_SQUARED_ * _JOULE_TO_KCAL_PER_MOL_ * _METER_TO_ANGSTROM_;

}   // namespace constants

#endif   // _CONSTANTS_HPP_