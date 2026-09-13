#include "emprise/hardware.hpp"
#include <thread>
#include <fstream>
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif
namespace emprise {
Hardware detect_hardware() {
    Hardware h;
    h.logical_threads=std::thread::hardware_concurrency();
    if(!h.logical_threads) h.logical_threads=1;
#ifdef _WIN32
    MEMORYSTATUSEX m{};m.dwLength=sizeof(m);
    if(GlobalMemoryStatusEx(&m)) { h.total_ram=m.ullTotalPhys;h.available_ram=m.ullAvailPhys; }
#else
    auto pages=sysconf(_SC_PHYS_PAGES),size=sysconf(_SC_PAGESIZE);
    if(pages>0 && size>0) h.total_ram=uint64_t(pages)*uint64_t(size);
    std::ifstream mem("/proc/meminfo");
    std::string key,line; uint64_t amount;
    while(mem>>key>>amount) { std::getline(mem,line);if(key=="MemAvailable:") { h.available_ram=amount*1024;break; } }
#endif
    return h;
}
}
