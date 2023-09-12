#include "qmSettings.hpp"

using settings::QMSettings;

/**
 * @brief sets the qmMethod to enum in settings
 *
 * @param method
 */
void QMSettings::setQMMethod(const std::string_view &method)
{
    if ("dftbplus" == method)
        _qmMethod = QMMethod::DFTBPLUS;
    else
        _qmMethod = QMMethod::NONE;
}