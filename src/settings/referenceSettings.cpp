#include "referenceSettings.hpp"

/**
 * @brief Adds a reference file and its corresponding BibTeX file.
 *
 * @param referenceFileName The name of the reference file to add.
 */
void ReferenceSettings::addReferenceFile(const std::string& referenceFileName)
{
    _referenceFileNames.insert(referenceFileName);
    _bibtexFileNames.insert(referenceFileName + ".bib");
}

/**
 * @brief Retrieves the set of reference file names.
 *
 * @return A constant reference to the set of reference file names.
 */
const std::unordered_set<std::string>& ReferenceSettings::getReferenceFileNames(
) const
{
    return _referenceFileNames;
}

/**
 * @brief Retrieves the set of BibTeX file names.
 *
 * @return A constant reference to the set of BibTeX file names.
 */
const std::unordered_set<std::string>& ReferenceSettings::getBibtexFileNames(
) const
{
    return _bibtexFileNames;
}
