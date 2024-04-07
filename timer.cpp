#include "timer.hpp"

Timer::Timer() : totalTime(0.0), isRunning(false) {}

void Timer::start() {
    if (!isRunning) {
        startTime = std::chrono::high_resolution_clock::now();
        isRunning = true;
    }
}

void Timer::stop() {
    if (isRunning) {
        std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
        totalTime += std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime);
        isRunning = false;
    }
}

void Timer::reset() {
    isRunning = false;
    totalTime = std::chrono::duration<double>(0.0);
}

double Timer::getTotalTime() const {
    return totalTime.count();
}