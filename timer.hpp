#ifndef TIMER_HPP_INCLUDED
#define TIMER_HPP_INCLUDED

#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::duration<double> totalTime;
    bool isRunning;

public:
    Timer();

    void start();

    void stop();

    void reset();

    double getTotalTime() const;
};

#endif  // TIMER_HPP_INCLUDED
