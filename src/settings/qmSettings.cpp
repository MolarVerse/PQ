#include "qmSettings.hpp"

#include "stringUtilities.hpp"   // for toLowerCopy

using settings::QMMethod;
using settings::QMSettings;

/**
 * @brief returns the qmMethod as string
 *
 * @param method
 * @return std::string
 */
std::string settings::string(const QMMethod method)
{
    switch (method)
    {
    case QMMethod::DFTBPLUS: return "DFTBPLUS";
    case QMMethod::PYSCF: return "PYSCF";
    default: return "none";
    }
}

/**
 * @brief sets the qmMethod to enum in settings
 *
 * @param method
 */
void QMSettings::setQMMethod(const std::string_view &method)
{
    const auto methodToLower = utilities::toLowerCopy(method);

    if ("dftbplus" == method)
        _qmMethod = QMMethod::DFTBPLUS;
    else if ("pyscf" == method)
        _qmMethod = QMMethod::PYSCF;
    else
        _qmMethod = QMMethod::NONE;
}