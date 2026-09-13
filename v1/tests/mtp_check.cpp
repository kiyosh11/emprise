// Diagnostic: measures MTP draft acceptance under different input conventions.
#include "emprise/model.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: mtp-check MODEL [steps]");
        emprise::ModelOptions options;
        emprise::Model model(argv[1], options);
        auto tokens = model.tokenizer().encode(
            model.tokenizer().chat_prompt("Explain how a refrigerator works."), true);
        auto logits = model.prefill(tokens);
        auto h_prev = model.last_hidden();
        auto h_prev_raw = model.last_hidden_raw();
        const int steps = argc > 2 ? std::stoi(argv[2]) : 32;
        auto argmax = [](const std::vector<float>& v) {
            return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
        };
        std::array<int, 4> match{};
        long long position = static_cast<long long>(tokens.size());
        for (int s = 0; s < steps; ++s) {
            const int t1 = argmax(logits);
            auto logits1 = model.evaluate(t1, true);
            auto h_cur = model.last_hidden();
            auto h_cur_raw = model.last_hidden_raw();
            const int t2 = argmax(logits1);
            auto test = [&](const std::vector<float>& h, int tok) {
                model.reset_mtp();
                return argmax(model.mtp_draft(h, tok, static_cast<size_t>(position))) == t2;
            };
            match[0] += test(h_prev_raw, t1);
            match[1] += test(h_prev, t1);
            match[2] += test(h_cur_raw, t1);
            match[3] += test(h_cur, t1);
            logits = std::move(logits1);
            h_prev = std::move(h_cur);
            h_prev_raw = std::move(h_cur_raw);
            ++position;
        }
        std::cout << "steps " << steps << "\n"
                  << "A (raw h_{L-1}, t1):   " << match[0] << "\n"
                  << "B (norm h_{L-1}, t1):  " << match[1] << "\n"
                  << "C (raw h_L, t1):       " << match[2] << "\n"
                  << "D (norm h_L, t1):      " << match[3] << "\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
