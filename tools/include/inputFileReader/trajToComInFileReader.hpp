#ifndef _TRAJTOCOM_INFILEREADER_HPP_

#define _TRAJTOCOM_INFILEREADER_HPP_

#include "inputFileReader.hpp"

class TrajToComInFileReader : public InputFileReader
{
public:
    explicit TrajToComInFileReader(const std::string_view &filename) : InputFileReader(filename) {}

    AnalysisRunner &read() override;
};

#endif // _TRAJECTORYTOCOM_INPUTINPUTFILEREADER_HPP_