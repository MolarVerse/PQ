#ifndef _PARAMETER_FILE_READER_HPP_

#define _PARAMETER_FILE_READER_HPP_

#include "engine.hpp"
#include "parameterFileSection.hpp"

#include <string>

namespace readInput::parameterFile
{
    class ParameterFileReader;
    void readParameterFile(engine::Engine &);

}   // namespace readInput::parameterFile

/**
 * @class ParameterReader
 *
 * @brief reads parameter file and sets settings
 *
 */
class readInput::parameterFile::ParameterFileReader
{
  private:
    std::string     _filename;
    std::ifstream   _fp;
    engine::Engine &_engine;

    std::vector<std::unique_ptr<readInput::parameterFile::ParameterFileSection>> _parameterFileSections;

  public:
    ParameterFileReader(const std::string &filename, engine::Engine &engine);

    bool                                            isNeeded() const;
    void                                            read();
    readInput::parameterFile::ParameterFileSection *determineSection(const std::vector<std::string> &);

    void setFilename(const std::string_view &filename) { _filename = filename; }
};

#endif   // _PARAMETER_FILE_READER_HPP_