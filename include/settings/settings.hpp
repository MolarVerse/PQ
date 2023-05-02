#ifndef _SETTINGS_H_

#define _SETTINGS_H_

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