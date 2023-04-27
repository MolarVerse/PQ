#include <string>
#include <memory>
#include <iostream>

#include "rstFileReader.hpp"

using namespace std;

RstFileReader::RstFileReader(string) {}
RstFileReader::~RstFileReader() {}

// void read(string filename);

unique_ptr<SimulationBox> read_rst(string filename)
{
    cout << filename << endl;
    auto sim = unique_ptr<SimulationBox>(new SimulationBox);
    return sim;
}