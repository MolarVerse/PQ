#include "qmmmSettings.hpp"

using settings::QMMMSettings;

/********************
 *                  *
 * standard setters *
 *                  *
 ********************/

/**
 * @brief set the qmCenter string in the settings
 *
 * @param qmCenter
 */
void QMMMSettings::setQMCenterString(const std::string_view qmCenter)
{
    _qmCenterString = qmCenter;
}

/**
 * @brief set the qmOnlyList string in the settings
 *
 * @param qmOnlyList
 */
void QMMMSettings::setQMOnlyListString(const std::string_view qmOnlyList)
{
    _qmOnlyListString = qmOnlyList;
}

/**
 * @brief set the mmOnlyList string in the settings
 *
 * @param mmOnlyList
 */
void QMMMSettings::setMMOnlyListString(const std::string_view mmOnlyList)
{
    _mmOnlyListString = mmOnlyList;
}

/**
 * @brief set the useQMCharges in the settings
 *
 * @param useQMCharges
 */
void QMMMSettings::setUseQMCharges(const bool useQMCharges)
{
    _useQMCharges = useQMCharges;
}

/**
 * @brief set the qmCoreRadius in the settings
 *
 * @param qmCoreRadius
 */
void QMMMSettings::setQMCoreRadius(const double qmCoreRadius)
{
    _qmCoreRadius = qmCoreRadius;
}

/**
 * @brief set the qmmmLayerRadius in the settings
 *
 * @param qmmmLayerRadius
 */
void QMMMSettings::setQMMMLayerRadius(const double qmmmLayerRadius)
{
    _qmmmLayerRadius = qmmmLayerRadius;
}

/**
 * @brief set the qmmmSmoothingRadius in the settings
 *
 * @param qmmmSmoothingRadius
 */
void QMMMSettings::setQMMMSmoothingRadius(const double qmmmSmoothingRadius)
{
    _qmmmSmoothingRadius = qmmmSmoothingRadius;
}

/********************
 *                  *
 * standard getters *
 *                  *
 ********************/

/**
 * @brief get the qmCenter string
 *
 * @return std::string
 */
std::string QMMMSettings::getQMCenterString() { return _qmCenterString; }

/**
 * @brief get the qmOnlyList string
 *
 * @return std::string
 */
std::string QMMMSettings::getQMOnlyListString() { return _qmOnlyListString; }

/**
 * @brief get the mmOnlyList string
 *
 * @return std::string
 */
std::string QMMMSettings::getMMOnlyListString() { return _mmOnlyListString; }

/**
 * @brief get the useQMCharges
 *
 * @return bool
 */
bool QMMMSettings::getUseQMCharges() { return _useQMCharges; }

/**
 * @brief get the qmCoreRadius
 *
 * @return double
 */
double QMMMSettings::getQMCoreRadius() { return _qmCoreRadius; }

/**
 * @brief get the qmmmLayerRadius
 *
 * @return double
 */
double QMMMSettings::getQMMMLayerRadius() { return _qmmmLayerRadius; }

/**
 * @brief get the qmmmSmoothingRadius
 *
 * @return double
 */
double QMMMSettings::getQMMMSmoothingRadius() { return _qmmmSmoothingRadius; }