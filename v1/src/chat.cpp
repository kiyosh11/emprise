#include "emprise/model.hpp"
#include "json.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#ifdef _OPENMP
#include <omp.h>
#endif
int main(int argc,char** argv) {
    try {
        std::string model,prompt="Explain how a refrigerator works.",file,report_file;
        size_t count=32;
        emprise::ModelOptions options;
        bool tokenize=false,raw=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            auto value=[&] { if(i+1>=argc) throw std::invalid_argument("Missing value for "+arg); return std::string(argv[++i]); };
            if(arg=="--model") model=value();
            else if(arg=="--prompt") prompt=value();
            else if(arg=="--prompt-file") file=value();
            else if(arg=="--report") report_file=value();
            else if(arg=="--tokens") count=std::stoull(value());
            else if(arg=="--context") options.context=std::stoull(value());
            else if(arg=="--ram-mib") options.weight_ram_budget=std::stoull(value())*1024*1024;
            else if(arg=="--vram-mib") options.weight_vram_budget=std::stoull(value())*1024*1024;
            else if(arg=="--cpu") options.cuda=false;
            else if(arg=="--fast-q8") options.quantize_activations=true;
            else if(arg=="--unfused-ffn") options.fused_ffn=false;
            else if(arg=="--tokenize") tokenize=true;
            else if(arg=="--raw") raw=true;
            else if(arg=="--disk") {
                auto v=value();
                if(v=="disabled") options.disk=emprise::DiskPolicy::disabled;
                else if(v=="auto") options.disk=emprise::DiskPolicy::automatic;
                else if(v=="enabled") options.disk=emprise::DiskPolicy::enabled;
                else throw std::invalid_argument("Unknown disk policy");
            } else if(arg=="--threads") {
                int threads=std::stoi(value());
                if(threads<1) throw std::invalid_argument("Invalid thread count");
                options.threads=static_cast<unsigned>(threads);
            } else throw std::invalid_argument("Unknown option: "+arg);
        }
        if(model.empty()) throw std::invalid_argument("Usage: emprise-chat --model MODEL.gguf [--prompt TEXT] [--cpu] [--tokens N] [--tokenize]");
        if(!file.empty()) { std::ifstream f(file,std::ios::binary); if(!f) throw std::runtime_error("Cannot read prompt"); prompt.assign(std::istreambuf_iterator<char>(f),{}); }
        if(tokenize) {
            emprise::Gguf g(model); emprise::Tokenizer t(g.metadata_json());
            auto tokens=t.encode(raw?prompt:t.chat_prompt(prompt),true);
            std::cout << '[';
            for(size_t i=0;i<tokens.size();++i) std::cout << (i?",":"") << tokens[i];
            std::cout << "]\n"; return 0;
        }
        auto start=std::chrono::steady_clock::now();
        emprise::Model engine(model,options);
        std::cerr << "Backend: " << engine.backend_name() << '\n';
        auto tokens=engine.tokenizer().encode(raw?prompt:engine.tokenizer().chat_prompt(prompt),true);
        auto loaded=std::chrono::steady_clock::now(),first=loaded,last=loaded;
        size_t emitted=0;
        std::string output_text;
        std::vector<int> output_tokens;
        std::vector<double> arrivals;
        engine.generate(tokens,count,[&](int t) {
            last=std::chrono::steady_clock::now(); if(!emitted) first=last; ++emitted;
            auto piece=engine.tokenizer().decode(t);output_text+=piece;output_tokens.push_back(t);
            arrivals.push_back(std::chrono::duration<double>(last-loaded).count());
            std::cout << piece << std::flush; return true;
        });
        std::cout << '\n';
        auto seconds=[](auto d){ return std::chrono::duration<double>(d).count(); };
        std::cerr << "Load: " << seconds(loaded-start) << "s; first token after load: " << seconds(first-loaded)
                  << "s; generated: " << emitted << "; decode: " << (emitted>1?(emitted-1)/seconds(last-first):0) << " tokens/s\n";
        auto profile=engine.profile();
        std::cerr << "Matrix work (includes first uploads and synchronization): " << profile.matrix_seconds
                  << "s in " << profile.matrix_calls << " calls; other work: " << profile.non_matrix_seconds << "s\n";
        std::cerr << "CPU non-matrix: norm " << profile.norm_seconds << "s, attention " << profile.attention_seconds
                  << "s, recurrent " << profile.recurrent_seconds << "s\n";
        if(!report_file.empty()) {
            nlohmann::json report={{"engine","emprise custom prototype"},{"backend",engine.backend_name()},
                {"model",model},{"prompt_tokens",tokens},{"output_tokens",output_tokens},{"text",output_text},
                {"load_seconds",seconds(loaded-start)},{"first_token_seconds",seconds(first-loaded)},
                {"generation_tokens_per_second",emitted>1?(emitted-1)/seconds(last-first):0},
                {"content_arrivals_seconds",arrivals},{"matrix_seconds",profile.matrix_seconds},
                {"matrix_calls",profile.matrix_calls},{"non_matrix_seconds",profile.non_matrix_seconds},
                {"context",options.context},{"quantized_activations",options.quantize_activations},{"fused_ffn",options.fused_ffn}};
            std::ofstream f(report_file);
            if(!f) throw std::runtime_error("Cannot write report");
            f<<report.dump(2,' ',false,nlohmann::json::error_handler_t::replace)<<'\n';
        }
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n';return 1; }
}
