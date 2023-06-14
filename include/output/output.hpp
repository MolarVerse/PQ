#ifndef _OUTPUT_H_

#define _OUTPUT_H_

#include <string>
#include <fstream>

/**
 * @class Output
 *
 * @brief Base class for output files
 *
 */
class Output
{
protected:
    std::string _filename;
    std::ofstream _fp;
    inline static size_t _outputFrequency = 1;
    int _rank;

    void openFile();

public:
    explicit Output(const std::string &filename) : _filename(filename){};

    void setFilename(const std::string_view &filename);

    static void setOutputFrequency(const size_t outputFreq);

    std::string initialMomentumMessage(const double momentum) const;

    // standard getter and setters
    std::string getFilename() const { return _filename; };

    static size_t getOutputFrequency() { return _outputFrequency; };
};

/**
 * @class LogOutput inherits from Output
 *
 * @brief Output file for log file
 *
 */
class LogOutput : public Output
{
public:
    using Output::Output;

    void writeDensityWarning();
    void writeRelaxationTimeThermostatWarning();
    void writeInitialMomentum(const double momentum);
};

/**
 * @class StdoutOutput inherits from Output
 *
 * @brief Output file for stdout
 *
 */
class StdoutOutput : public Output
{
public:
    using Output::Output;

    void writeDensityWarning() const;
    void writeRelaxationTimeThermostatWarning() const;
    void writeInitialMomentum(const double momentum) const;
};

/**
 * @class RstFileOutput inherits from Output
 *
 * @brief Output file for restart file
 *
 */
class RstFileOutput : public Output
{
public:
    using Output::Output;
};

/**
 * @class ChargeOutput inherits from Output
 *
 * @brief Output file for charge file
 *
 */
class ChargeOutput : public Output
{
public:
    using Output::Output;
};

#endif