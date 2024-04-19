#include "ioutils.cuh"

void savebin(const string& filename, const void* gpudata, uint size)
{
    // Copy data from GPU to CPU
    void* data = malloc(size);
    cudaMemcpy(data, gpudata, size, cudaMemcpyDeviceToHost);

    // Write data to file
    FILE* file = fopen(filename.c_str(), "wb");
    fwrite(data, 1, size, file);
    fclose(file);
    
    // Free memory
    free(data);

}

uint findsize(const string& filename)
{
    // Read data from file
    FILE* file = fopen(filename.c_str(), "rb");
    fseek(file, 0, SEEK_END);
    uint size = ftell(file);
    fclose(file);
    
    return size;
}

void loadbin(const string& filename, void* gpudata, uint size)
{
    // Allocate memory
    void* data = malloc(size);

    // Read data from file
    FILE* file = fopen(filename.c_str(), "rb");
    fread(data, 1, size, file);
    fclose(file);
    
    // Copy data from CPU to GPU
    cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);

    // Free memory
    free(data);

}