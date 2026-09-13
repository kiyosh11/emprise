// Diagnostic: runs the model once with the device-resident feed-forward chain
// enabled and once with the separate operations, then compares the traces
// directly. This isolates the fused path from any reference-engine difference.
#include "emprise/model.hpp"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

using Vec = std::vector<float>;
using Trace = std::map<std::string, Vec>;

Trace run(const std::string& path, const std::vector<int>& tokens, bool fused, bool quantize) {
    emprise::ModelOptions options;
    options.fused_ffn = fused;
    options.quantize_activations = quantize;
    Trace captured;
    options.trace = [&](size_t pos, const std::string& name, std::span<const float> x) {
        captured[std::to_string(pos) + "." + name] = Vec(x.begin(), x.end());
    };
    emprise::Model model(path, options);
    for (int token : tokens) model.evaluate(token);
    return captured;
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("Usage: fused-check MODEL TOKENS.json [fast]");
        std::ifstream input(argv[2]);
        nlohmann::json ids;
        input >> ids;
        auto tokens = ids.get<std::vector<int>>();
        const bool quantize = argc > 3 && std::string(argv[3]) == "fast";
        auto fused = run(argv[1], tokens, true, quantize);
        auto separate = run(argv[1], tokens, false, quantize);
        if (fused.size() != separate.size())
            throw std::runtime_error("Trace size mismatch: fused=" + std::to_string(fused.size()) +
                                     " separate=" + std::to_string(separate.size()));
        std::string worst;
        double worst_relative = 0, total_energy = 0, total_error = 0;
        for (const auto& [name, a] : fused) {
            auto found = separate.find(name);
            if (found == separate.end()) throw std::runtime_error("Missing separate trace: " + name);
            const auto& b = found->second;
            if (a.size() != b.size()) throw std::runtime_error("Length mismatch: " + name);
            double error = 0, energy = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                double d = double(a[i]) - b[i];
                error += d * d;
                energy += double(b[i]) * b[i];
            }
            double relative = std::sqrt(error / std::max(energy, 1e-30));
            total_error += error;
            total_energy += energy;
            if (relative > worst_relative) { worst_relative = relative; worst = name; }
        }
        std::cout.precision(6);
        for (const auto& [name, a] : fused) {
            if (name.starts_with("0.l_out-") || name == "0.result_norm" || name == "0.logits" ||
                name == "1.logits" || name == "31.logits" ||
                name == "31.l_out-31" || name == "31.result_norm") {
                const auto& b = separate.at(name);
                double error = 0, energy = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    double d = double(a[i]) - b[i];
                    error += d * d;
                    energy += double(b[i]) * b[i];
                }
                std::cout << name << " relative_l2 " << std::sqrt(error / std::max(energy, 1e-30)) << '\n';
            }
        }
        std::cout << "overall relative_l2 " << std::sqrt(total_error / std::max(total_energy, 1e-30))
                  << " worst " << worst << " " << worst_relative << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
