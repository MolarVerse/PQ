#ifndef _OUTPUT_H_

#define _OUTPUT_H_

#include <string>

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
    static int _outputFreq;

public:
    std::string getFilename() const { return _filename; };
    void setFilename(std::string_view filename);

    static int getOutputFreq() { return _outputFreq; };
    static void setOutputFreq(int outputFreq);
};

/**
 * @class EnergyOutput inherits from Output
 *
 * @brief Output file for energy
 *
 */
class EnergyOutput : public Output
{
public:
    EnergyOutput() = default;
};

/**
 * @class TrajectoryOutput inherits from Output
 *
 * @brief Output file for xyz, vel, force files
 *
 */
class TrajectoryOutput : public Output
{
public:
    TrajectoryOutput() = default;
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
    LogOutput() = default;
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
    StdoutOutput() = default;
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
    RstFileOutput() = default;
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
    ChargeOutput() = default;
};

/**
 * @class InfoOutput inherits from Output
 *
 * @brief Output file for info file
 *
 */
class InfoOutput : public Output
{
public:
    InfoOutput() = default;
};

#endif