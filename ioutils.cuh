#include <iostream>
#include <string>
using namespace std;

void savebin(const string& filename, const void* gpudata, uint size);

uint findsize(const string& filename);

void loadbin(const string& filename, void* gpudata, uint size);