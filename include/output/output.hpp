#ifndef _OUTPUT_H_

#define _OUTPUT_H_

#include <fstream>
#include <string>

namespace output
{
    class Output;
}

/**
 * @class Output
 *
 * @brief Base class for output files
 *
 */
class output::Output
{
  protected:
    std::string          _filename;
    std::ofstream        _fp;
    inline static size_t _outputFrequency = 1;
    int                  _rank;

    void openFile();

  public:
    explicit Output(const std::string &filename) : _filename(filename){};

    void setFilename(const std::string_view &filename);

    static void setOutputFrequency(const size_t outputFreq);

    std::string initialMomentumMessage(const double momentum) const;

    // standard getter and setters
    std::string getFilename() const { return _filename; }

    static size_t getOutputFrequency() { return _outputFrequency; }
};

#endif