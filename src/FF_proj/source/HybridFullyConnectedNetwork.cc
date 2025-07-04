#ifndef HYBRID_FULLY_CONNECTED_NETWORK_CC
#define HYBRID_FULLY_CONNECTED_NETWORK_CC

// This program is generated by machine at a random time

#include "HybridFullyConnectedNetwork.h"
#include "GetVTable.h"
#include <algorithm>
#include <random>

GET_V_TABLE(HybridFullyConnectedNetwork)

float sigmoid1(float x) { return 1 / (1 + exp(-x)); }

float relu1(float x) { return std::max(0.0f, x); }

float outLabel1(float x) {
    float threshold = 0.5;
    if (x > threshold) {
        return 1;
    } else {
        return 0;
    }
}

void randomGen1(Handle<Vector<float>> vec) {

    std::random_device rd;

    std::mt19937 e2(rd());

    std::uniform_real_distribution<> distp(0.0001, 0.5);
    std::uniform_real_distribution<> distn(-0.5, -0.0001);

    auto gen = std::bind(std::uniform_int_distribution<>(0, 1),
                       std::default_random_engine());

    for (int i = 0; i < vec->size(); i++) {
        (*vec)[i] =  (bool)gen() ? distn(e2) : distp(e2);
    }
}

#endif
