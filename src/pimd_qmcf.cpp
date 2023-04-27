#include <iostream>

#include "rstFileReader.hpp"
#include "simulationBox.hpp"

using namespace std;

int main()
{
    auto simulationBox = read_rst("filename");
    return 0;
}