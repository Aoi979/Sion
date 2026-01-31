#pragma once
#include <vector>
#include <functional>
#include "compare.hpp"

struct TestRegistry {
    std::vector<std::pair<const char*, std::function<void()>>> tests;
    static TestRegistry& inst(){ static TestRegistry r; return r; }
};

struct TestAdder {
    TestAdder(const char* name, std::function<void()> fn){
        TestRegistry::inst().tests.emplace_back(name, fn);
    }
};

#define SION_TEST(name) \
    void name(); \
    static TestAdder _sion_test_adder_##name(#name, name); \
    void name()

#define SION_CHECK(x) \
    do { \
        if(!(x)){ \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " check failed: " #x << std::endl; \
            std::exit(1); \
        } \
    } while(0)
