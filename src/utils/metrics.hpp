#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <numeric>
#include <chrono>

namespace yolo {


struct LatencyTracker {
    std::vector<int64_t> latencies_us;

    void record(int64_t latency) {
        latencies_us.push_back(latency);
    }

    void printStats(const std::string& name) const {
        if(latencies_us.empty()) {
            std::cout << name << ": No data recorded.\n" << std::endl;
            return;
        }
        
        std::vector<int64_t> sorted_lat = latencies_us;
        std::sort(sorted_lat.begin(), sorted_lat.end());

        size_t n = sorted_lat.size();
        int64_t total_time = std::accumulate(sorted_lat.begin(), sorted_lat.end(), 0LL);
        double avg = static_cast<double>(total_time)/n;

        auto getPercentile = [&](double p) -> int64_t {
            size_t idx = static_cast<size_t>(n*p);
            if (idx >= n) idx = n-1;
            return sorted_lat[idx];
        };

        double fps = 1000000.0 / avg;

        std::cout << "\n===" << name << " Stats ===" << std::endl;
        std::cout << "Frames Processed: " << n << std::endl;
        std::cout << "Throughput (Avg FPS): " << fps << std::endl;
        std::cout << "Avg Latency:   " << avg << " us" << std::endl;
        std::cout << "Min Latency:   " << sorted_lat.front() << " us" << std::endl;
        std::cout << "Max Latency:   " << sorted_lat.back() << " us" << std::endl;
        std::cout << "p50 Latency:   " << getPercentile(0.50) << " us" << std::endl;
        std::cout << "p99 Latency:   " << getPercentile(0.99) << " us" << std::endl;
        std::cout << "p99.9 Latency: " << getPercentile(0.999) << " us" << std::endl;
        std::cout << "=========================" << std::endl;
    
    }



};



}