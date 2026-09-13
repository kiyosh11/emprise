#include "emprise/model.hpp"
#include "emprise/hardware.hpp"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace emprise {
namespace {
using Vec = std::vector<float>;
float sigmoid(float x) { return x >= 0 ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x)); }
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x))); }
void rms(std::span<float> x, std::span<const float> weight, float eps) {
    if (x.size() != weight.size()) throw std::invalid_argument("RMS normalization dimensions");
    float sum = 0;
    for (auto v : x) sum += v*v;
    float scale = 1.f / std::sqrt(sum / x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] *= scale * weight[i];
}
}
struct Model::Impl {
    Gguf weights;
    Tokenizer tok;
    ModelOptions options;
    std::unique_ptr<Backend> backend;
    int hidden, layers, heads, kv_heads, head_dim, rotary, key_heads, value_heads, state_dim, conv_width, eos;
    float eps, rope_base;
    size_t position = 0;
    Profile profile;
    std::unordered_map<std::string, Vec> small;
    struct LayerState { Vec conv, recurrent, keys, values; };
    std::vector<LayerState> states;
    std::vector<bool> linear_layers;
    LayerState mtp_state;   // the blk.32 MTP block's own attention cache
    Vec hidden_out;         // output-normalized hidden of the last evaluated position
    Vec hidden_raw_out;     // raw (pre-output-norm) hidden of the last evaluated position
    Vec mtp_hidden;         // hidden produced by the last mtp_step (for chaining)
    // device-resident buffers (only when the backend supports the device path)
    uint64_t dev_x=0,dev_norm=0,dev_tmp=0,dev_attn=0,dev_qkv=0,dev_logits=0;

    Impl(const std::filesystem::path& path, ModelOptions opts): weights(path), tok(weights.metadata_json()), options(opts) {
#ifdef _OPENMP
        // Spreading the per-head CPU work across every logical processor can hurt
        // on multi-socket machines, so cap the default well below that.
        const unsigned workers=options.threads?options.threads:
            std::min(16u,std::max(1u,std::thread::hardware_concurrency()));
        omp_set_num_threads(static_cast<int>(workers));
#endif
        auto m = nlohmann::json::parse(weights.metadata_json());
        if (m.value("general.architecture", "") != "qwen35") throw std::runtime_error("Only qwen35 is implemented");
        auto integer = [&](const char* key) { return m.at(std::string("qwen35.") + key).get<int>(); };
        hidden=integer("embedding_length");
        layers=integer("block_count")-m.value("qwen35.nextn_predict_layers",0);
        heads=integer("attention.head_count"); kv_heads=integer("attention.head_count_kv");
        head_dim=integer("attention.key_length"); rotary=integer("rope.dimension_count");
        key_heads=integer("ssm.group_count"); value_heads=integer("ssm.time_step_rank");
        state_dim=integer("ssm.state_size"); conv_width=integer("ssm.conv_kernel");
        eos=m.at("tokenizer.ggml.eos_token_id").get<int>();
        eps=m.at("qwen35.attention.layer_norm_rms_epsilon").get<float>();
        rope_base=m.at("qwen35.rope.freq_base").get<float>();
        if (std::min({hidden,layers,heads,kv_heads,head_dim,rotary,key_heads,value_heads,state_dim,conv_width}) <= 0 ||
            heads % kv_heads || value_heads % key_heads || rotary > head_dim || rotary % 2 ||
            options.context == 0 || options.context > static_cast<size_t>(integer("context_length")))
            throw std::runtime_error("Unsupported or invalid Qwen dimensions/context");
        if (!options.weight_ram_budget) options.weight_ram_budget=detect_hardware().available_ram/10*7;
        weights.configure_storage(options.disk, options.weight_ram_budget);
        backend = options.cuda ? cuda_backend(options.weight_vram_budget,options.quantize_activations) : cpu_backend();
        states.resize(layers);
        for(int i=0;i<layers;++i) {
            const auto name="blk."+std::to_string(i)+".ssm_a";
            linear_layers.push_back(std::any_of(weights.tensors().begin(),weights.tensors().end(),[&](const Tensor& t){return t.name==name;}));
        }
    }
    const Vec& parameter(const std::string& name) {
        auto it = small.find(name);
        if (it != small.end()) return it->second;
        const auto& t = weights.tensor(name);
        if (t.bytes > 16 * 1024 * 1024) throw std::runtime_error("Unexpected non-matrix parameter size");
        std::vector<std::byte> raw(static_cast<size_t>(t.bytes));
        weights.read(t, 0, raw);
        Vec values(static_cast<size_t>(t.elements)); dequantize(t.type, raw, values);
        return small.emplace(name, std::move(values)).first->second;
    }
    Vec mm(const std::string& name, const Vec& x) {
        const auto& t = weights.tensor(name);
        if (t.shape.size()!=2) throw std::runtime_error("Expected matrix: " + name);
        Vec y(static_cast<size_t>(t.shape[1]));
        auto start=std::chrono::steady_clock::now();
        backend->linear(weights,t,x,y);
        profile.matrix_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ++profile.matrix_calls;
        return y;
    }
    Vec norm(Vec x, const std::string& name) { rms(x,parameter(name),eps); return x; }
    std::vector<Vec> mms(const std::vector<std::string>& names, const Vec& x) {
        std::vector<Vec> outputs;
        outputs.reserve(names.size());
        for (const auto& name : names) {
            const auto& t = weights.tensor(name);
            if (t.shape.size() != 2 || t.shape[0] != x.size()) throw std::runtime_error("Expected matrix: " + name);
            outputs.emplace_back(static_cast<size_t>(t.shape[1]));
        }
        std::vector<LinearRequest> requests;
        requests.reserve(names.size());
        for (size_t i = 0; i < names.size(); ++i) requests.push_back({&weights.tensor(names[i]), outputs[i]});
        auto start = std::chrono::steady_clock::now();
        backend->linear_many(weights, requests, x);
        profile.matrix_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        profile.matrix_calls += names.size();
        return outputs;
    }
    void trace(const std::string& name, const Vec& x) {
        if(options.trace) options.trace(position,name,x);
    }
    void rope(std::span<float> x, int pos) {
        for (int i=0;i<rotary/2;++i) {
            float angle = static_cast<float>(pos) / std::pow(rope_base, 2.f*i/rotary);
            float a=x[i], b=x[i+rotary/2], c=std::cos(angle), s=std::sin(angle);
            x[i]=a*c-b*s; x[i+rotary/2]=a*s+b*c;
        }
    }
    void rope(std::span<float> x) { rope(x, static_cast<int>(position)); }
    Vec attention_preout(const std::string& p, Vec qg, Vec k, Vec v, LayerState& st, int pos) {
        if (qg.size()!=size_t(heads*head_dim*2) || k.size()!=size_t(kv_heads*head_dim) || v.size()!=k.size())
            throw std::runtime_error("Attention projection dimensions");
        auto& qn=parameter(p+"attn_q_norm.weight"); auto& kn=parameter(p+"attn_k_norm.weight");
        for(int h=0;h<heads;++h) { auto q=std::span(qg).subspan(h*head_dim*2,head_dim); rms(q,qn,eps); rope(q,pos); }
        for(int h=0;h<kv_heads;++h) { auto kh=std::span(k).subspan(h*head_dim,head_dim); rms(kh,kn,eps); rope(kh,pos); }
        st.keys.insert(st.keys.end(),k.begin(),k.end()); st.values.insert(st.values.end(),v.begin(),v.end());
        Vec out(heads*head_dim);
        const size_t stride=k.size(), length=st.keys.size()/stride;
        #pragma omp parallel for schedule(static)
        for(int h=0;h<heads;++h) {
            int kh=h/(heads/kv_heads);
            Vec scores(length);
            float peak=-INFINITY;
            for(size_t t=0;t<length;++t) {
                float dot=0;
                for(int d=0;d<head_dim;++d) dot+=qg[h*head_dim*2+d]*st.keys[t*stride+kh*head_dim+d];
                scores[t]=dot/std::sqrt(float(head_dim)); peak=std::max(peak,scores[t]);
            }
            float total=0;
            for(auto& a:scores) { a=std::exp(a-peak); total+=a; }
            for(int d=0;d<head_dim;++d) {
                float sum=0;
                for(size_t t=0;t<length;++t) sum+=scores[t]/total*st.values[t*stride+kh*head_dim+d];
                out[h*head_dim+d]=sum*sigmoid(qg[h*head_dim*2+head_dim+d]);
            }
        }
        return out;
    }
    Vec attention_from_qkv(const std::string& p, Vec qg, Vec k, Vec v, LayerState& st, int pos) {
        return mm(p+"attn_output.weight",attention_preout(p,std::move(qg),std::move(k),std::move(v),st,pos));
    }
    Vec attention(const std::string& p, const Vec& x, LayerState& st) {
        auto projections=mms({p+"attn_q.weight",p+"attn_k.weight",p+"attn_v.weight"},x);
        return attention_from_qkv(p,std::move(projections[0]),std::move(projections[1]),std::move(projections[2]),st,static_cast<int>(position));
    }
    Vec recurrent_from_projections(const std::string& p, Vec qkv, Vec gate, Vec alpha, Vec beta, LayerState& st, int slot) {
        auto& conv=parameter(p+"ssm_conv1d.weight"); auto& a=parameter(p+"ssm_a");
        auto& dt=parameter(p+"ssm_dt.bias"); auto& nw=parameter(p+"ssm_norm.weight");
        const int key_dim=key_heads*state_dim, value_dim=value_heads*state_dim, channels=2*key_dim+value_dim;
        if (qkv.size()!=size_t(channels) || gate.size()!=size_t(value_dim) ||
            alpha.size()!=size_t(value_heads) || beta.size()!=alpha.size() ||
            conv.size()!=size_t(channels*conv_width) || a.size()!=alpha.size() || dt.size()!=alpha.size() || nw.size()!=size_t(state_dim))
            throw std::runtime_error("Recurrent projection dimensions");
        if(st.conv.empty()) st.conv.resize(channels*conv_width);
        if(st.recurrent.empty()) st.recurrent.resize(size_t(value_heads)*state_dim*state_dim);
        Vec out(value_dim);
        Backend::DeltaNet d;
        d.key_heads=key_heads; d.value_heads=value_heads; d.state_dim=state_dim; d.conv_width=conv_width;
        d.eps=eps; d.slot=slot;
        d.qkv=qkv; d.gate=gate; d.alpha=alpha; d.beta=beta;
        d.conv=conv; d.ssm_a=a; d.dt=dt; d.norm=nw;
        d.conv_state=st.conv; d.rec_state=st.recurrent; d.out=out;
        backend->delta_net(weights,d);
        return mm(p+"ssm_out.weight",out);
    }
    Vec recurrent(const std::string& p, const Vec& x, LayerState& st, int slot) {
        auto& conv=parameter(p+"ssm_conv1d.weight"); auto& a=parameter(p+"ssm_a");
        auto& dt=parameter(p+"ssm_dt.bias"); auto& nw=parameter(p+"ssm_norm.weight");
        const int key_dim=key_heads*state_dim, value_dim=value_heads*state_dim, channels=2*key_dim+value_dim;
        if (conv.size()!=size_t(channels*conv_width) || a.size()!=size_t(value_heads) ||
            dt.size()!=a.size() || nw.size()!=size_t(state_dim))
            throw std::runtime_error("Recurrent parameter dimensions");
        if(st.conv.empty()) st.conv.resize(channels*conv_width);
        if(st.recurrent.empty()) st.recurrent.resize(size_t(value_heads)*state_dim*state_dim);
        Vec out(hidden);
        Backend::RecurrentBlock b;
        b.qkv_w=&weights.tensor(p+"attn_qkv.weight"); b.gate_w=&weights.tensor(p+"attn_gate.weight");
        b.alpha_w=&weights.tensor(p+"ssm_alpha.weight"); b.beta_w=&weights.tensor(p+"ssm_beta.weight");
        b.out_w=&weights.tensor(p+"ssm_out.weight");
        b.conv=conv; b.ssm_a=a; b.dt=dt; b.norm=nw;
        b.key_heads=key_heads; b.value_heads=value_heads; b.state_dim=state_dim; b.conv_width=conv_width;
        b.eps=eps; b.slot=slot;
        b.x=x; b.conv_state=st.conv; b.rec_state=st.recurrent; b.out=out;
        backend->recurrent_block(weights,b);
        return out;
    }
    Vec evaluate(int token, bool logits) {
        const auto start=std::chrono::steady_clock::now();
        const double matrix_start=profile.matrix_seconds;
        if(position>=options.context) throw std::runtime_error("Context budget exhausted");
        const auto& emb=weights.tensor("token_embd.weight");
        if(token<0 || static_cast<uint64_t>(token)>=emb.shape.at(1)) throw std::out_of_range("Token ID");
        Vec x(hidden); std::vector<std::byte> row(static_cast<size_t>(encoded_bytes(emb.type,hidden)));
        weights.read(emb,static_cast<uint64_t>(token)*row.size(),row); dequantize(emb.type,row,x);
        for(int layer=0;layer<layers;++layer) {
            const auto p="blk."+std::to_string(layer)+".";
            auto norm_start=std::chrono::steady_clock::now();
            auto normalized=norm(x,p+"attn_norm.weight");
            profile.norm_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-norm_start).count();
            trace("attn_norm-"+std::to_string(layer),normalized);
            // Infer actual layer type from its tensors, rather than a fixed interval.
            const auto wall_start=std::chrono::steady_clock::now();
            const double matrix_before=profile.matrix_seconds;
            auto attn=linear_layers[layer]?recurrent(p,normalized,states[layer],layer):attention(p,normalized,states[layer]);
            double cpu_time=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count()-(profile.matrix_seconds-matrix_before);
            (linear_layers[layer]?profile.recurrent_seconds:profile.attention_seconds)+=cpu_time;
            for(int i=0;i<hidden;++i) x[i]+=attn[i];
            trace("attn_residual-"+std::to_string(layer),x);
            normalized=norm(x,p+"post_attention_norm.weight");
            Vec down;
            if(options.fused_ffn) {
                down.resize(hidden);
                auto ff_start=std::chrono::steady_clock::now();
                backend->feed_forward(weights,weights.tensor(p+"ffn_gate.weight"),weights.tensor(p+"ffn_up.weight"),
                                      weights.tensor(p+"ffn_down.weight"),normalized,down);
                profile.matrix_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-ff_start).count();
                profile.matrix_calls+=3;
            } else {
            auto gate=mm(p+"ffn_gate.weight",normalized), up=mm(p+"ffn_up.weight",normalized);
            if(gate.size()!=up.size()) throw std::runtime_error("FFN dimensions");
            for(size_t i=0;i<gate.size();++i) gate[i]=silu(gate[i])*up[i];
            down=mm(p+"ffn_down.weight",gate);
            }
            for(int i=0;i<hidden;++i) x[i]+=down[i];
            trace("l_out-"+std::to_string(layer),x);
        }
        Vec result;
        if(logits) {
            hidden_raw_out=x;
            x=norm(std::move(x),"output_norm.weight");
            trace("result_norm",x);
            hidden_out=x;
            result=mm("output.weight",x);
            trace("logits",result);
        }
        ++position;
        profile.non_matrix_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-(profile.matrix_seconds-matrix_start);
        ++profile.evaluated_tokens;
        return result;
    }
    Vec project_flat(const std::string& name, std::span<const float> X, int K) {
        const auto& t = weights.tensor(name);
        if (t.shape.size()!=2 || X.size()!=static_cast<size_t>(t.shape[0])*static_cast<size_t>(K))
            throw std::runtime_error("Expected matrix: " + name);
        Vec y(static_cast<size_t>(K)*static_cast<size_t>(t.shape[1]));
        auto start=std::chrono::steady_clock::now();
        backend->linear_batch(weights,t,X,y,K);
        profile.matrix_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        ++profile.matrix_calls;
        return y;
    }
    // Processes K tokens together. The projections are batched so each weight
    // read is shared across the tokens; the causal attention and the recurrent
    // scan still run position by position.
    std::vector<Vec> evaluate_block(const std::vector<int>& tokens, int logits) {
        const int K=static_cast<int>(tokens.size());
        if(K<1) throw std::invalid_argument("Empty block");
        if(position+static_cast<size_t>(K)>options.context) throw std::runtime_error("Context budget exhausted");
        const auto& emb=weights.tensor("token_embd.weight");
        std::vector<Vec> h(K);
        for(int k=0;k<K;++k) {
            if(tokens[k]<0 || static_cast<uint64_t>(tokens[k])>=emb.shape.at(1)) throw std::out_of_range("Token ID");
            h[k].resize(hidden);
            std::vector<std::byte> row(static_cast<size_t>(encoded_bytes(emb.type,hidden)));
            weights.read(emb,static_cast<uint64_t>(tokens[k])*row.size(),row);
            dequantize(emb.type,row,h[k]);
        }
        const int base=static_cast<int>(position);
        for(int layer=0;layer<layers;++layer) {
            const auto p="blk."+std::to_string(layer)+".";
            std::vector<Vec> normalized(K);
            for(int k=0;k<K;++k) normalized[k]=norm(h[k],p+"attn_norm.weight");
            Vec X(static_cast<size_t>(K)*hidden);
            for(int k=0;k<K;++k) std::copy(normalized[k].begin(),normalized[k].end(),X.begin()+static_cast<size_t>(k)*hidden);
            std::vector<Vec> attn(K);
            if(linear_layers[layer]) {
                Vec qkv=project_flat(p+"attn_qkv.weight",X,K);
                Vec gate=project_flat(p+"attn_gate.weight",X,K);
                Vec alpha=project_flat(p+"ssm_alpha.weight",X,K);
                Vec beta=project_flat(p+"ssm_beta.weight",X,K);
                const size_t c=qkv.size()/K, g=gate.size()/K, av=alpha.size()/K;
                for(int k=0;k<K;++k) {
                    Vec q(qkv.begin()+k*c,qkv.begin()+(k+1)*c);
                    Vec ga(gate.begin()+k*g,gate.begin()+(k+1)*g);
                    Vec al(alpha.begin()+k*av,alpha.begin()+(k+1)*av);
                    Vec be(beta.begin()+k*av,beta.begin()+(k+1)*av);
                    attn[k]=recurrent_from_projections(p,std::move(q),std::move(ga),std::move(al),std::move(be),states[layer],layer);
                }
            } else {
                Vec q=project_flat(p+"attn_q.weight",X,K);
                Vec kk=project_flat(p+"attn_k.weight",X,K);
                Vec vv=project_flat(p+"attn_v.weight",X,K);
                const size_t qd=q.size()/K, kd=kk.size()/K;
                for(int k=0;k<K;++k) {
                    Vec qq(q.begin()+k*qd,q.begin()+(k+1)*qd);
                    Vec kx(kk.begin()+k*kd,kk.begin()+(k+1)*kd);
                    Vec vx(vv.begin()+k*kd,vv.begin()+(k+1)*kd);
                    attn[k]=attention_from_qkv(p,std::move(qq),std::move(kx),std::move(vx),states[layer],base+k);
                }
            }
            for(int k=0;k<K;++k) for(int i=0;i<hidden;++i) h[k][i]+=attn[k][i];
            for(int k=0;k<K;++k) normalized[k]=norm(h[k],p+"post_attention_norm.weight");
            for(int k=0;k<K;++k) std::copy(normalized[k].begin(),normalized[k].end(),X.begin()+static_cast<size_t>(k)*hidden);
            Vec gate=project_flat(p+"ffn_gate.weight",X,K);
            Vec up=project_flat(p+"ffn_up.weight",X,K);
            Vec sw(gate.size());
            for(size_t i=0;i<sw.size();++i) sw[i]=silu(gate[i])*up[i];
            Vec down=project_flat(p+"ffn_down.weight",sw,K);
            for(int k=0;k<K;++k) for(int i=0;i<hidden;++i) h[k][i]+=down[static_cast<size_t>(k)*hidden+i];
        }
        std::vector<Vec> result;
        if(logits) {
            const int first=logits==1?K-1:0;
            const int n=K-first;
            Vec X(static_cast<size_t>(n)*hidden);
            for(int k=first;k<K;++k) {
                Vec hn=norm(h[k],"output_norm.weight");
                if(k==K-1) { hidden_raw_out=h[k]; hidden_out=hn; }
                std::copy(hn.begin(),hn.end(),X.begin()+static_cast<std::ptrdiff_t>(k-first)*hidden);
            }
            const auto& ow=weights.tensor("output.weight");
            Vec Y(static_cast<size_t>(n)*ow.shape[1]);
            auto last=std::chrono::steady_clock::now();
            backend->linear_batch(weights,ow,X,Y,n);
            profile.matrix_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-last).count();
            ++profile.matrix_calls;
            result.resize(static_cast<size_t>(n));
            for(int k=0;k<n;++k)
                result[k]=Vec(Y.begin()+static_cast<std::ptrdiff_t>(k)*ow.shape[1],
                              Y.begin()+static_cast<std::ptrdiff_t>(k+1)*ow.shape[1]);
        }
        position+=K;
        profile.evaluated_tokens+=K;
        return result;
    }
    std::vector<float> prefill(const std::vector<int>& tokens) {
        if(tokens.empty()) throw std::invalid_argument("Empty prompt");
        std::vector<float> logits;
        for(size_t i=0;i<tokens.size();) {
            const size_t n=std::min<size_t>(4,tokens.size()-i);
            std::vector<int> chunk(tokens.begin()+static_cast<std::ptrdiff_t>(i),tokens.begin()+static_cast<std::ptrdiff_t>(i+n));
            auto out=evaluate_block(chunk, i+n==tokens.size()?1:0);
            if(!out.empty()) logits=std::move(out.back());
            i+=n;
        }
        return logits;
    }
    Vec embed(int token) {
        const auto& emb=weights.tensor("token_embd.weight");
        if(token<0 || static_cast<uint64_t>(token)>=emb.shape.at(1)) throw std::out_of_range("Token ID");
        Vec x(hidden);
        std::vector<std::byte> row(static_cast<size_t>(encoded_bytes(emb.type,hidden)));
        weights.read(emb,static_cast<uint64_t>(token)*row.size(),row);
        dequantize(emb.type,row,x);
        return x;
    }
    // One MTP draft step for the blk.32 next-n block: given the previously
    // produced hidden and token, returns the logits for the following token.
    Vec mtp_step(const Vec& hidden_in, int token, int pos) {
        Vec en=norm(embed(token),"blk.32.nextn.enorm.weight");
        Vec hn=norm(Vec(hidden_in.begin(),hidden_in.end()),"blk.32.nextn.hnorm.weight");
        Vec concat(static_cast<size_t>(2)*hidden);
        std::copy(en.begin(),en.end(),concat.begin());
        std::copy(hn.begin(),hn.end(),concat.begin()+static_cast<std::ptrdiff_t>(hidden));
        Vec cur=mm("blk.32.nextn.eh_proj.weight",concat);
        Vec a=norm(cur,"blk.32.attn_norm.weight");
        auto projections=mms({"blk.32.attn_q.weight","blk.32.attn_k.weight","blk.32.attn_v.weight"},a);
        Vec attn=attention_from_qkv("blk.32.",std::move(projections[0]),std::move(projections[1]),std::move(projections[2]),mtp_state,pos);
        for(int i=0;i<hidden;++i) cur[i]+=attn[i];
        Vec post=norm(cur,"blk.32.post_attention_norm.weight");
        auto gate=mm("blk.32.ffn_gate.weight",post), up=mm("blk.32.ffn_up.weight",post);
        for(size_t i=0;i<gate.size();++i) gate[i]=silu(gate[i])*up[i];
        Vec down=mm("blk.32.ffn_down.weight",gate);
        for(int i=0;i<hidden;++i) cur[i]+=down[i];
        mtp_hidden=cur;
        Vec head_n=norm(std::move(cur),"blk.32.nextn.shared_head_norm.weight");
        return mm("output.weight",head_n);
    }
    bool use_device() const {
        if(!backend->device_path()) return false;
        uint64_t need=0;
        for(const auto& t:weights.tensors()) {
            if(t.name.rfind("blk.32.",0)==0) continue;
            need += (t.type==WeightType::q6_k)? (t.elements/256)*216 : (t.type==WeightType::q4_k)? (t.elements/256)*160 : t.bytes;
        }
        return need + 256ull*1024*1024 <= backend->device_capacity();
    }
    void ensure_dev() {
        if(dev_x) return;
        dev_x=backend->dev_alloc(static_cast<size_t>(hidden)*sizeof(float));
        dev_norm=backend->dev_alloc(static_cast<size_t>(hidden)*sizeof(float));
        dev_tmp=backend->dev_alloc(static_cast<size_t>(hidden)*sizeof(float));
        dev_attn=backend->dev_alloc(static_cast<size_t>(heads*head_dim)*sizeof(float));
        dev_qkv=backend->dev_alloc(static_cast<size_t>(heads*head_dim*2+2*kv_heads*head_dim)*sizeof(float));
    }
    ~Impl() {
        if(backend) {
            for(uint64_t p:{dev_x,dev_norm,dev_tmp,dev_attn,dev_qkv,dev_logits}) backend->dev_free(p);
        }
    }
    // Full token forward with the residual stream kept in VRAM. Only the
    // attention hop (q/k/v down, attention output up) touches the host.
    Vec evaluate_device(int token,bool want_logits) {
        if(position>=options.context) throw std::runtime_error("Context budget exhausted");
        ensure_dev();
        Vec x=embed(token);
        backend->dev_upload(dev_x,x.data(),x.size()*sizeof(float));
        for(int layer=0;layer<layers;++layer) {
            const auto p="blk."+std::to_string(layer)+".";
            backend->dev_rmsnorm(weights,dev_x,weights.tensor(p+"attn_norm.weight"),dev_norm,hidden,eps);
            if(linear_layers[layer]) {
                Backend::RecurrentBlock b;
                b.qkv_w=&weights.tensor(p+"attn_qkv.weight"); b.gate_w=&weights.tensor(p+"attn_gate.weight");
                b.alpha_w=&weights.tensor(p+"ssm_alpha.weight"); b.beta_w=&weights.tensor(p+"ssm_beta.weight");
                b.out_w=&weights.tensor(p+"ssm_out.weight");
                b.conv=parameter(p+"ssm_conv1d.weight"); b.ssm_a=parameter(p+"ssm_a");
                b.dt=parameter(p+"ssm_dt.bias"); b.norm=parameter(p+"ssm_norm.weight");
                b.key_heads=key_heads; b.value_heads=value_heads; b.state_dim=state_dim; b.conv_width=conv_width;
                b.eps=eps; b.slot=layer;
                backend->dev_recurrent(weights,b,dev_norm,dev_tmp);
            } else {
                const int qd=heads*head_dim*2, kd=kv_heads*head_dim;
                backend->dev_linear(weights,weights.tensor(p+"attn_q.weight"),dev_norm,dev_qkv);
                backend->dev_linear(weights,weights.tensor(p+"attn_k.weight"),dev_norm,dev_qkv+static_cast<uint64_t>(qd)*sizeof(float));
                backend->dev_linear(weights,weights.tensor(p+"attn_v.weight"),dev_norm,dev_qkv+static_cast<uint64_t>(qd+kd)*sizeof(float));
                Vec qkv(static_cast<size_t>(qd+2*kd));
                backend->dev_download(qkv.data(),dev_qkv,qkv.size()*sizeof(float));
                Vec qg(qkv.begin(),qkv.begin()+qd);
                Vec kx(qkv.begin()+qd,qkv.begin()+qd+kd);
                Vec vx(qkv.begin()+qd+kd,qkv.end());
                Vec attn=attention_preout(p,std::move(qg),std::move(kx),std::move(vx),states[layer],static_cast<int>(position));
                backend->dev_upload(dev_attn,attn.data(),attn.size()*sizeof(float));
                backend->dev_linear(weights,weights.tensor(p+"attn_output.weight"),dev_attn,dev_tmp);
            }
            backend->dev_add(dev_x,dev_tmp,hidden);
            backend->dev_rmsnorm(weights,dev_x,weights.tensor(p+"post_attention_norm.weight"),dev_norm,hidden,eps);
            backend->dev_ffn(weights,weights.tensor(p+"ffn_gate.weight"),weights.tensor(p+"ffn_up.weight"),weights.tensor(p+"ffn_down.weight"),dev_norm,dev_tmp);
            backend->dev_add(dev_x,dev_tmp,hidden);
        }
        Vec result;
        if(want_logits) {
            backend->dev_rmsnorm(weights,dev_x,weights.tensor("output_norm.weight"),dev_norm,hidden,eps);
            const auto& ow=weights.tensor("output.weight");
            const int vocab=static_cast<int>(ow.shape[1]);
            if(!dev_logits) dev_logits=backend->dev_alloc(static_cast<size_t>(vocab)*sizeof(float));
            backend->dev_linear(weights,ow,dev_norm,dev_logits);
            result.resize(static_cast<size_t>(vocab));
            backend->dev_download(result.data(),dev_logits,result.size()*sizeof(float));
        }
        ++position;
        ++profile.evaluated_tokens;
        return result;
    }
};
Model::Model(const std::filesystem::path& p, ModelOptions o):impl_(std::make_unique<Impl>(p,o)) {}
Model::~Model()=default;
const Tokenizer& Model::tokenizer() const { return impl_->tok; }
std::string Model::backend_name() const { return impl_->backend->name(); }
Profile Model::profile() const { return impl_->profile; }
void Model::reset() { impl_->position=0; impl_->states.clear(); impl_->states.resize(impl_->layers); impl_->backend->reset_state(); impl_->profile={}; }
std::vector<float> Model::evaluate(int token,bool logits) { return impl_->evaluate(token,logits); }
std::vector<std::vector<float>> Model::evaluate_block(const std::vector<int>& tokens) { return impl_->evaluate_block(tokens,2); }
std::vector<float> Model::prefill(const std::vector<int>& tokens) { return impl_->prefill(tokens); }
std::vector<float> Model::last_hidden() const { return impl_->hidden_out; }
std::vector<float> Model::last_hidden_raw() const { return impl_->hidden_raw_out; }
std::vector<float> Model::mtp_draft(const std::vector<float>& hidden, int token, size_t pos) { return impl_->mtp_step(hidden, token, static_cast<int>(pos)); }
std::vector<float> Model::mtp_hidden() const { return impl_->mtp_hidden; }
void Model::reset_mtp() { impl_->mtp_state = Impl::LayerState{}; }
void Model::generate(const std::vector<int>& prompt,size_t count,const std::function<bool(int)>& output,const std::atomic_bool* cancelled) {
    if(prompt.empty()) throw std::invalid_argument("Empty prompt");
    if(prompt.size()>impl_->options.context || count>impl_->options.context-prompt.size())
        throw std::invalid_argument("Prompt plus output exceeds context budget");
    reset();
    if(!count) return;
    if(cancelled && cancelled->load()) return;
    const bool dev=impl_->use_device();
    Vec logits=impl_->prefill(prompt);
    for(size_t i=0;i<count;++i) {
        if(cancelled && cancelled->load()) return;
        int token=static_cast<int>(std::max_element(logits.begin(),logits.end())-logits.begin());
        if(token==impl_->eos || !output(token)) return;
        if(i+1<count) logits = dev? impl_->evaluate_device(token,true) : impl_->evaluate(token,true);
    }
}
}
