#ifndef _TOOLENGINE_HPP_

#define _TOOLENGINE_HPP_

#include <string>
#include <memory>
#include <map>
#include <functional>

#include "analysisRunner.hpp"
#include "inputFileReader.hpp"

class Engine
{
private:
    std::string _executableName;
    std::string _inputFilename;

    std::map<std::string, std::function<void(const std::string_view &)>> _analysisRunnerKeys;

    std::vector<InputFileReader *> _inputFileReaders;
    std::vector<AnalysisRunner *> _analysisRunners;

public:
    Engine(const std::string_view &, const std::string_view &);
    ~Engine()
    {
        for (auto inputFileReader : _inputFileReaders)
            delete inputFileReader;
    }

    void run();
    void addAnalysisRunnerKeys();
    void parseAnalysisRunners();
};

#endif // _ENGINE_HPP_