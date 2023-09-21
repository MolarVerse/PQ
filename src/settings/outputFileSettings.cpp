#include "outputFileSettings.hpp"

#include <cstdint>   // for UINT64_MAX

using settings::OutputFileSettings;

/**
 * @brief Sets the output frequency of the simulation
 *
 * @param outputFreq
 *
 * @throw InputFileException if output frequency is negative
 *
 * @note
 *  if output frequency is 0, it is set to UINT64_MAX
 *  in order to avoid division by 0 in the output
 *
 */
void OutputFileSettings::setOutputFrequency(const size_t outputFreq)
{
    if (0 == outputFreq)
        _outputFrequency = UINT64_MAX;
    else
        _outputFrequency = outputFreq;
}