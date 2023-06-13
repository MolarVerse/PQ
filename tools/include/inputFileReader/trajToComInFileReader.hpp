#ifndef _TRAJTOCOM_INFILEREADER_HPP_

#define _TRAJTOCOM_INFILEREADER_HPP_

#include "inputFileReader.hpp"
#include "trajToCom.hpp"

class TrajToComInFileReader : public InputFileReader
{
private:
    TrajToCom _runner;

public:
    using InputFileReader::InputFileReader;

    AnalysisRunner &read() override;
};

#endif // _TRAJECTORYTOCOM_INPUTINPUTFILEREADER_HPP_