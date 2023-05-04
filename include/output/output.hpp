#ifndef _OUTPUT_H_

#define _OUTPUT_H_

#include <string>

class Output
{
protected:
    std::string _filename;
    static int _outputFreq;

public:
    std::string getFilename() const { return _filename; };
    void setFilename(std::string_view filename) { _filename = filename; }; // TODO: check if this file exists

    static int getOutputFreq() { return _outputFreq; };
    static void setOutputFreq(int outputFreq);
};

class EnergyOutput : public Output
{
public:
    EnergyOutput() = default;
};

class TrajectoryOutput : public Output
{
public:
    TrajectoryOutput() = default;
};

class LogOutput : public Output
{
public:
    LogOutput() = default;
};

class StdoutOutput : public Output
{
    // TODO: think of a clever way to add stdout stream
public:
    explicit StdoutOutput() = default;
};

class RstFileOutput : public Output
{
public:
    RstFileOutput() = default;
};

class ChargeOutput : public Output
{
public:
    ChargeOutput() = default;
};

class InfoOutput : public Output
{
public:
    InfoOutput() = default;
};

#endif