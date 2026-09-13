#pragma once
#include <cstdint>
namespace emprise {
struct Hardware { uint64_t total_ram=0, available_ram=0; unsigned logical_threads=1; };
Hardware detect_hardware();
}
