// Diagnostic: the batched forward (evaluate_block) must produce the same logits
// as running the model token by token. Compares both on an identical token
// sequence in separate model instances.
#include "emprise/model.hpp"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("Usage: block-check MODEL TOKENS.json [block]");
        std::ifstream input(argv[2]);
        nlohmann::json ids;
        input >> ids;
        auto all = ids.get<std::vector<int>>();
        const size_t block = argc > 3 ? std::stoul(argv[3]) : 32;
        std::vector<int> tokens(all.begin(), all.begin() + std::min(block, all.size()));
        emprise::ModelOptions options;
        options.fused_ffn = false; // match the batched path's separate projections
        std::vector<std::vector<float>> sequential;
        double sequential_seconds = 0;
        {
            emprise::Model model(argv[1], options);
            model.evaluate(0, true); // upload weights including the output head
            model.reset();
            auto start = std::chrono::steady_clock::now();
            for (int token : tokens) sequential.push_back(model.evaluate(token, true));
            sequential_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        std::vector<std::vector<float>> batched;
        double batched_seconds = 0;
        {
            emprise::Model model(argv[1], options);
            model.evaluate(0, true); // upload weights including the output head
            model.reset();
            auto start = std::chrono::steady_clock::now();
            batched = model.evaluate_block(tokens);
            batched_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        if (sequential.size() != batched.size()) throw std::runtime_error("Position count mismatch");
        auto argmax = [](const std::vector<float>& v) {
            return static_cast<size_t>(std::max_element(v.begin(), v.end()) - v.begin());
        };
        double worst = 0;
        size_t worst_position = 0;
        int mismatches = 0;
        for (size_t k = 0; k < sequential.size(); ++k) {
            double error = 0, energy = 0;
            for (size_t i = 0; i < sequential[k].size(); ++i) {
                double d = double(batched[k][i]) - sequential[k][i];
                error += d * d;
                energy += double(sequential[k][i]) * sequential[k][i];
            }
            const double relative = std::sqrt(error / std::max(energy, 1e-30));
            if (relative > worst) { worst = relative; worst_position = k; }
            if (argmax(sequential[k]) != argmax(batched[k])) ++mismatches;
        }
        std::cout.precision(6);
        std::cout << "positions " << sequential.size() << " worst relative L2 " << worst
                  << " at position " << worst_position << " argmax mismatches " << mismatches << '\n';
        std::cout << "sequential " << sequential_seconds << "s (" << sequential.size() / sequential_seconds
                  << " tok/s), batched " << batched_seconds << "s (" << sequential.size() / batched_seconds
                  << " tok/s), speedup " << sequential_seconds / batched_seconds << "x\n";
        if (worst > 1e-3 || mismatches) throw std::runtime_error("Batched forward mismatch");
        std::cout << "Batched forward matches the sequential path\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
