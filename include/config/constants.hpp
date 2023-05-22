#ifndef _CONSTANTS_H_

#define _CONSTANTS_H_

/**
 * @brief Conversion factors for mass units
 */
constexpr double _AMU_TO_KG_ = 1.660539040e-27;
constexpr double _KG_TO_AMU_ = 1.0 / _AMU_TO_KG_;
constexpr double _AVOGADRO_NUMBER_ = _KG_TO_AMU_ / 1000;

/**
 * @brief Conversion factors for length units
 */
constexpr double _ANGSTROM_TO_METER_ = 1.0e-10;
constexpr double _METER_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_METER_;

constexpr double _BOHR_RADIUS_TO_METER_ = 5.2917721067e-11;
constexpr double _METER_TO_BOHR_RADIUS_ = 1.0 / _BOHR_RADIUS_TO_METER_;
constexpr double _ANGSTROM_TO_BOHR_RADIUS_ = _ANGSTROM_TO_METER_ / _BOHR_RADIUS_TO_METER_;
constexpr double _BOHR_RADIUS_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_BOHR_RADIUS_;

/**
 * @brief Conversion factors for volume units
 */
constexpr double _ANGSTROM_CUBIC_TO_METER_CUBIC_ = _ANGSTROM_TO_METER_ * _ANGSTROM_TO_METER_ * _ANGSTROM_TO_METER_;
constexpr double _METER_CUBIC_TO_ANGSTROM_CUBIC_ = 1.0 / _ANGSTROM_CUBIC_TO_METER_CUBIC_;
constexpr double _ANGSTROM_CUBIC_TO_LITER_ = _ANGSTROM_CUBIC_TO_METER_CUBIC_ * 1.0e3;
constexpr double _LITER_TO_ANGSTROM_CUBIC_ = 1.0 / _ANGSTROM_CUBIC_TO_LITER_;

/**
 * @brief Conversion factors for density units
 */
constexpr double _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_ = _KG_TO_AMU_ / _LITER_TO_ANGSTROM_CUBIC_;
constexpr double _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_ = 1.0 / _KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_;

/**
 * @brief Conversion factors for energy units
 */
constexpr double _KCAL_TO_JOULE_ = 4184.0;
constexpr double _JOULE_TO_KCAL_ = 1.0 / _KCAL_TO_JOULE_;

/**
 * @brief Conversion factors for electron related data
 */
constexpr double _ELECTRON_CHARGE_ = 1.6021766208e-19; // in Coulomb
constexpr double _ELECTRON_CHARGE_SQUARED_ = _ELECTRON_CHARGE_ * _ELECTRON_CHARGE_;
constexpr double _ELECTRON_MASS_ = 9.10938356e-31; // in kg

/**
 * @brief Conversion factors for time units
 */
constexpr double _S_TO_FS_ = 1.0e15;
constexpr double _FS_TO_S_ = 1.0 / _S_TO_FS_;

/**
 * @brief Conversion factors for velocity verler integrator
 *
 * @details
 * [F] = kcal/mol/Angstrom
 * [m] = amu
 * [v] = Angstrom/s
 * [t] = fs
 *
 * [v] = [t] * [F] / [m]
 *
 * [F'] = J/m = [F] * 4.184 * 1000 / 6.022140857e23 / 1e-10 = [F] * _KCAL_TO_JOULE_ / _AVOGADRO_NUMBER / _ANGSTROM_TO_METER_
 * [m'] = kg = [m] * 1.660539040e-27 = [m] * _AMU_TO_KG_
 *
 */
constexpr double _FORCE_UNIT_TO_SI_ = _KCAL_TO_JOULE_ / _AVOGADRO_NUMBER_ / _ANGSTROM_TO_METER_;
constexpr double _MASS_UNIT_TO_SI_ = _AMU_TO_KG_;
constexpr double _TIME_UNIT_TO_SI_ = _FS_TO_S_;

constexpr double _VELOCITY_UNIT_TO_SI_ = _ANGSTROM_TO_METER_;
constexpr double _SI_TO_VELOCITY_UNIT_ = 1.0 / _VELOCITY_UNIT_TO_SI_;

constexpr double _V_VERLET_VELOCITY_FACTOR_ = 0.5 * _FORCE_UNIT_TO_SI_ / _MASS_UNIT_TO_SI_ * _TIME_UNIT_TO_SI_ * _SI_TO_VELOCITY_UNIT_;

#endif