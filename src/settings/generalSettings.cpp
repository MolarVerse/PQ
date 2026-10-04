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

#include "generalSettings.hpp"

#include <string>
#include <utility>

namespace settings
{

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief sets the jobtype to enum in settings
     *
     * @param jobtype
     */
    void GeneralSettings::setJobtype(JobType jobtype)
    {
        _jobtype = jobtype;

        switch (jobtype)
        {
            using enum JobType;

            case MM_OPT:       // fallthrough
            case MM_HESSIAN:   // fallthrough
            case MM_MD:        // fallthrough
            case QM_MD:        // fallthrough
            case QMMM_MD:      // fallthrough
            case NONE: deactivateRingPolymerMD(); break;
            case RING_POLYMER_QM_MD: activateRingPolymerMD(); break;
        }
    }

    /**
     * @brief sets the floating point type
     *
     * @param floatingPointType
     */
    void GeneralSettings::setFloatingPointType(FPType floatingPointType)
    {
        _floatingPointType = floatingPointType;
    }

    /**
     * @brief sets the random seed value
     *
     * @param randomSeed
     */
    void GeneralSettings::setRandomSeed(uint_fast32_t randomSeed)
    {
        _randomSeed = randomSeed;
    }

    /**
     * @brief sets if the random seed value has been set
     *
     * @param isRandomSeedSet
     */
    void GeneralSettings::setIsRandomSeedSet(bool isRandomSeedSet)
    {
        _isRandomSeedset = isRandomSeedSet;
    }

    /**
     * @brief sets Ring Polymer MD to active
     *
     * @param isRingPolymerMD
     */
    void GeneralSettings::setIsRingPolymerMDActivated(bool isRingPolymerMD)
    {
        _isRingPolymerMDActivated = isRingPolymerMD;
    }

    /**
     * @brief sets the dimensionality
     *
     * @param dimensionality
     */
    void GeneralSettings::setDimensionality(size_t dimensionality)
    {
        _dimensionality = dimensionality;
    }

    /**
     * @brief sets the virial type
     *
     * @param virialType
     */
    void GeneralSettings::setVirialType(VirialType virialType)
    {
        _virial = virialType;
    }

    /**
     * @brief sets the integrator type
     *
     * @param integratorType
     */
    void GeneralSettings::setIntegratorType(IntegratorType integratorType)
    {
        _integrator = integratorType;
    }

    /**
     * @brief sets the number of cells for the cell list
     *
     * @param numberOfCells
     */
    void GeneralSettings::setNumberOfCells(size_t numberOfCells)
    {
        _numberOfCells = numberOfCells;
    }

    /**
     * @brief gets the number of cells for the cell list
     *
     * @return size_t
     */
    size_t GeneralSettings::getNumberOfCells() { return _numberOfCells; }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief get the jobtype
     *
     * @return JobType
     */
    JobType GeneralSettings::getJobtype() { return _jobtype; }

    /**
     * @brief get the floating point type
     *
     * @return FPType
     */
    FPType GeneralSettings::getFloatingPointType()
    {
        return _floatingPointType;
    }

    /**
     * @brief get the floating point string representation used in pybind11
     * bindings
     *
     */
    std::string GeneralSettings::getFloatingPointPybindString()
    {
        if (_floatingPointType == FPType::FLOAT)
            return "float32";

        return "float64";
    }

    /**
     * @brief get the random seed value
     *
     * @return uint_fast32_t
     */
    uint_fast32_t GeneralSettings::getRandomSeed() { return _randomSeed; }

    /**
     * @brief get if the random seed value has been set
     *
     * @return bool
     */
    bool GeneralSettings::isRandomSeedSet() { return _isRandomSeedset; }

    /**
     * @brief get the dimensionality
     *
     * @return size_t
     */
    size_t GeneralSettings::getDimensionality() { return _dimensionality; }

    /**
     * @brief get the virial type
     *
     * @return VirialType
     */
    VirialType GeneralSettings::getVirialType() { return _virial; }

    /**
     * @brief get the integrator type
     *
     * @return IntegratorType
     */
    IntegratorType GeneralSettings::getIntegratorType() { return _integrator; }

    /******************************
     *                            *
     * standard is-active methods *
     *                            *
     ******************************/

    /**
     * @brief Returns true if the jobtype does not use any MM type simulations
     *
     * @return true/false if the jobtype does not use any MM type simulations
     *
     */
    bool GeneralSettings::isQMOnlyJobtype()
    {
        using enum JobType;

        switch (_jobtype)
        {
            case MM_MD:
            case QMMM_MD:
            case MM_OPT:
            case MM_HESSIAN:
            case NONE: return false;
            case QM_MD:
            case RING_POLYMER_QM_MD: return true;
        }

        std::unreachable();
    }

    /**
     * @brief Returns true if the jobtype does not use any QM type simulations
     *
     * @return true/false if the jobtype does not use any QM type simulations
     *
     */
    bool GeneralSettings::isMMOnlyJobtype()
    {
        return _jobtype == JobType::MM_MD;
    }

    /**
     * @brief Returns true if the jobtype is a hybrid type simulation
     *
     * @return true/false if the jobtype is a hybrid type simulation
     *
     */
    bool GeneralSettings::isHybridJobtype()
    {
        return _jobtype == JobType::QMMM_MD;
    }

    /**
     * @brief Returns true if the jobtype performs an MD simulation
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isMDJobType()
    {
        using enum JobType;

        auto isMD = false;
        isMD      = isMD || _jobtype == MM_MD;
        isMD      = isMD || _jobtype == QM_MD;
        isMD      = isMD || _jobtype == QMMM_MD;
        isMD      = isMD || _jobtype == RING_POLYMER_QM_MD;

        return isMD;
    }

    /**
     * @brief Returns true if the jobtype does is based on optimization
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isOptJobType() { return _jobtype == JobType::MM_OPT; }

    /**
     * @brief Returns true if the MM simulations are activated
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isMMActivated()
    {
        using enum JobType;

        auto isMM = false;

        isMM = isMM || _jobtype == MM_MD;
        isMM = isMM || _jobtype == QMMM_MD;
        isMM = isMM || _jobtype == MM_OPT;
        isMM = isMM || _jobtype == MM_HESSIAN;

        return isMM;
    }

    /**
     * @brief Returns true if the QM simulations are activated
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isQMActivated()
    {
        using enum JobType;

        auto isQM = false;

        isQM = isQM || _jobtype == QM_MD;
        isQM = isQM || _jobtype == QMMM_MD;
        isQM = isQM || _jobtype == RING_POLYMER_QM_MD;

        return isQM;
    }

    /**
     * @brief Returns true if only QM simulations are activated
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isQMOnlyActivated()
    {
        return isQMActivated() && !isMMActivated();
    }

    /**
     * @brief Returns true if only MM simulations are activated
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isMMOnlyActivated()
    {
        return isMMActivated() && !isQMActivated();
    }

    /**
     * @brief Returns true if the ring polymer MD simulations are activated
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isRingPolymerMDActivated()
    {
        return _isRingPolymerMDActivated;
    }

    /**
     * @brief Returns true if the cell list is activated
     *
     * @return true/false
     *
     */
    bool GeneralSettings::isCellListActivated() { return _isCellListActivated; }

    /*****************************
     *                           *
     * standard activate methods *
     *                           *
     *****************************/

    /**
     * @brief activate ring polymer MD simulations
     *
     */
    void GeneralSettings::activateRingPolymerMD()
    {
        _isRingPolymerMDActivated = true;
    }

    /**
     * @brief deactivate ring polymer MD simulations
     *
     */
    void GeneralSettings::deactivateRingPolymerMD()
    {
        _isRingPolymerMDActivated = false;
    }

    /**
     * @brief activate cell list
     *
     */
    void GeneralSettings::activateCellList() { _isCellListActivated = true; }

    /**
     * @brief deactivate cell list
     *
     */
    void GeneralSettings::deactivateCellList() { _isCellListActivated = false; }

}   // namespace settings
