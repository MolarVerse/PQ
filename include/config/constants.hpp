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
constexpr double _KG_PER_LITER_CUBIC_TO_AMU_PER_ANGSTROM_CUBIC_ = _KG_TO_AMU_ * _METER_CUBIC_TO_ANGSTROM_CUBIC_ * 1.0e3;
constexpr double _AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_ = 1.0 / _KG_PER_LITER_CUBIC_TO_AMU_PER_ANGSTROM_CUBIC_;

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

#endif