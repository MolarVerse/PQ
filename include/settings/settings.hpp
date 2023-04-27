#ifndef _SETTINGS_H_

#define _SETTINGS_H_

class Settings
{
private:
    int _stepcount = 0;

public:
    Settings();
    ~Settings();
    int getStepCount();
    void setStepCount(int);
};

#endif