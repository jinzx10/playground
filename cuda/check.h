#ifndef CHECK_H
#define CHECK_H

#include <cstdio>

#define CHECK(call, msg)                                        \
{                                                               \
    const cudaError_t error = call;                             \
    if (error != cudaSuccess) {                                 \
        printf("ERROR: %s: %s\n",                               \
                msg, cudaGetErrorString(error));                \
        exit(1);                                                \
    }                                                           \
}


#endif
