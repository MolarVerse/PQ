#ifndef _SETTINGS_H_

#define _SETTINGS_H_

/**
 * @class Settings
 * 
 * @brief Stores the settings of the simulation
 * 
 */
class Settings
{
private:
    int _stepcount = 0;

public:
    Settings() = default;
    ~Settings() = default;

    int getStepCount() const;
    void setStepCount(int);
};

#endif