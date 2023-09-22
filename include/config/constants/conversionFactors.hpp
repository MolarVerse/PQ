#ifndef _CONVERSION_FACTORS_HPP_

#define _CONVERSION_FACTORS_HPP_

#include "natureConstants.hpp"

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
    static constexpr double _GRAM_TO_KG_ = 1.0e-3;
    static constexpr double _KG_TO_GRAM_ = 1.0 / _GRAM_TO_KG_;
    static constexpr double _AMU_TO_KG_  = _GRAM_TO_KG_ / _AVOGADRO_NUMBER_;
    static constexpr double _KG_TO_AMU_  = 1.0 / _AMU_TO_KG_;

    /**
     * @brief Conversion factors for length units
     */
    static constexpr double _ANGSTROM_TO_METER_ = 1.0e-10;
    static constexpr double _METER_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_METER_;

    static constexpr double _BOHR_RADIUS_TO_METER_ = _BOHR_RADIUS_IN_METER_;
    static constexpr double _METER_TO_BOHR_RADIUS_ = 1.0 / _BOHR_RADIUS_TO_METER_;

    static constexpr double _ANGSTROM_TO_BOHR_RADIUS_ = _ANGSTROM_TO_METER_ / _BOHR_RADIUS_TO_METER_;
    static constexpr double _BOHR_RADIUS_TO_ANGSTROM_ = 1.0 / _ANGSTROM_TO_BOHR_RADIUS_;

    /**
     * @brief Conversion factors for area units
     */
    static constexpr double _ANGSTROM_SQUARED_TO_METER_SQUARED_ = _ANGSTROM_TO_METER_ * _ANGSTROM_TO_METER_;
    static constexpr double _METER_SQUARED_TO_ANGSTROM_SQUARED_ = 1 / _ANGSTROM_SQUARED_TO_METER_SQUARED_;

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
    static constexpr double _KCAL_TO_JOULE_                      = 4184.0;
    static constexpr double _JOULE_TO_KCAL_                      = 1.0 / _KCAL_TO_JOULE_;
    static constexpr double _JOULE_TO_KCAL_PER_MOL_              = _JOULE_TO_KCAL_ * _AVOGADRO_NUMBER_;
    static constexpr double _KCAL_PER_MOL_TO_JOULE_              = 1.0 / _JOULE_TO_KCAL_PER_MOL_;
    static constexpr double _HARTREE_TO_JOULE_PER_MOL_           = 2625.5002;
    static constexpr double _HARTREE_TO_KCAL_PER_MOL_            = _HARTREE_TO_JOULE_PER_MOL_ * _JOULE_TO_KCAL_;
    static constexpr double _BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ = _BOLTZMANN_CONSTANT_ * _JOULE_TO_KCAL_PER_MOL_;

    /**
     * @brief Conversion factors for squared energy units
     */
    static constexpr double _BOLTZMANN_CONSTANT_SQUARED_      = _BOLTZMANN_CONSTANT_ * _BOLTZMANN_CONSTANT_;
    static constexpr double _REDUCED_PLANCK_CONSTANT_SQUARED_ = _REDUCED_PLANCK_CONSTANT_ * _REDUCED_PLANCK_CONSTANT_;

    /**
     * @brief Conversion factors for force units
     */
    static constexpr double _HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_ =
        _HARTREE_TO_KCAL_PER_MOL_ / _BOHR_RADIUS_TO_ANGSTROM_;

    /**
     * @brief Conversion factors for time units
     */
    static constexpr double _S_TO_FS_  = 1.0e15;
    static constexpr double _FS_TO_S_  = 1.0 / _S_TO_FS_;
    static constexpr double _PS_TO_FS_ = 1.0e3;
    static constexpr double _FS_TO_PS_ = 1.0 / _PS_TO_FS_;

    /**
     * @brief Conversion factors for pressure calculation
     */
    static constexpr double _PASCAL_TO_BAR_ = 1.0e-5;
    static constexpr double _BAR_TO_PASCAL_ = 1.0 / _PASCAL_TO_BAR_;

    /**
     * @brief Conversion factors for velocities
     */
    static constexpr double _M_PER_S_TO_CM_PER_S_        = 1.0e2;
    static constexpr double _SPEED_OF_LIGHT_IN_CM_PER_S_ = _SPEED_OF_LIGHT_ * _M_PER_S_TO_CM_PER_S_;

    /**
     * @brief Conversion factors for frequencies
     */
    static constexpr double _WAVE_NUMBER_IN_PER_CM_TO_FREQUENCY_IN_HZ_ = _SPEED_OF_LIGHT_IN_CM_PER_S_;

}   // namespace constants

#endif   // _CONVERSION_FACTORS_HPP_