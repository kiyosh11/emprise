#include "emprise/backend.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace emprise {
using Vec = std::vector<float>;
namespace {
float sigmoid(float x) { return x >= 0 ? 1.f / (1.f + std::exp(-x)) : std::exp(x) / (1.f + std::exp(x)); }
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return std::max(x, 0.f) + std::log1p(std::exp(-std::abs(x))); }
void rms(std::span<float> x, std::span<const float> weight, float eps) {
    float sum = 0;
    for (auto v : x) sum += v*v;
    float scale = 1.f / std::sqrt(sum / x.size() + eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] *= scale * weight[i];
}
}
void Backend::delta_net(Gguf&, const DeltaNet& p) {
    const int key_dim=p.key_heads*p.state_dim, value_dim=p.value_heads*p.state_dim, channels=2*key_dim+value_dim;
    std::vector<float> qkv(p.qkv.begin(),p.qkv.end());
    for(int c=0;c<channels;++c) {
        auto* history=p.conv_state.data()+static_cast<size_t>(c)*p.conv_width;
        for(int j=0;j<p.conv_width-1;++j) history[j]=history[j+1];
        history[p.conv_width-1]=qkv[c];
        float sum=0;
        for(int j=0;j<p.conv_width;++j) sum+=history[j]*p.conv[c*p.conv_width+j];
        qkv[c]=silu(sum);
    }
    for(int h=0;h<p.key_heads;++h) {
        float qs=0,ks=0;
        for(int d=0;d<p.state_dim;++d) { float q=qkv[h*p.state_dim+d],k=qkv[key_dim+h*p.state_dim+d]; qs+=q*q;ks+=k*k; }
        float qscale=1.f/std::sqrt(qs+p.eps)/std::sqrt(float(p.state_dim)),kscale=1.f/std::sqrt(ks+p.eps);
        for(int d=0;d<p.state_dim;++d) { qkv[h*p.state_dim+d]*=qscale; qkv[key_dim+h*p.state_dim+d]*=kscale; }
    }
    #pragma omp parallel for schedule(static)
    for(int h=0;h<p.value_heads;++h) {
        const int kh=h%p.key_heads;
        const float* q=qkv.data()+kh*p.state_dim; const float* k=qkv.data()+key_dim+kh*p.state_dim;
        const float* v=qkv.data()+2*key_dim+h*p.state_dim;
        float* state=p.rec_state.data()+static_cast<size_t>(h)*p.state_dim*p.state_dim;
        const float decay=std::exp(p.ssm_a[h]*softplus(p.alpha[h]+p.dt[h])), b=sigmoid(p.beta[h]);
        for(int d=0;d<p.state_dim;++d) {
            float memory=0;
            #if !defined(_MSC_VER)
            #pragma omp simd reduction(+:memory)
            #endif
            for(int j=0;j<p.state_dim;++j) { state[d*p.state_dim+j]*=decay; memory+=state[d*p.state_dim+j]*k[j]; }
            const float delta=(v[d]-memory)*b;
            float sum=0;
            #if !defined(_MSC_VER)
            #pragma omp simd reduction(+:sum)
            #endif
            for(int j=0;j<p.state_dim;++j) { state[d*p.state_dim+j]+=k[j]*delta; sum+=state[d*p.state_dim+j]*q[j]; }
            p.out[h*p.state_dim+d]=sum;
        }
        auto chunk=p.out.subspan(static_cast<size_t>(h)*p.state_dim,p.state_dim); rms(chunk,p.norm,p.eps);
        for(int d=0;d<p.state_dim;++d) chunk[d]*=silu(p.gate[h*p.state_dim+d]);
    }
}
void Backend::recurrent_block(Gguf& model, const RecurrentBlock& p) {
    Vec qkv(static_cast<size_t>(p.qkv_w->shape[1])), gate(static_cast<size_t>(p.gate_w->shape[1]));
    Vec alpha(static_cast<size_t>(p.alpha_w->shape[1])), beta(static_cast<size_t>(p.beta_w->shape[1]));
    linear(model,*p.qkv_w,p.x,qkv);
    linear(model,*p.gate_w,p.x,gate);
    linear(model,*p.alpha_w,p.x,alpha);
    linear(model,*p.beta_w,p.x,beta);
    Vec dn(static_cast<size_t>(p.out_w->shape[0]));
    DeltaNet d;
    d.key_heads=p.key_heads; d.value_heads=p.value_heads; d.state_dim=p.state_dim; d.conv_width=p.conv_width;
    d.eps=p.eps; d.slot=p.slot;
    d.qkv=qkv; d.gate=gate; d.alpha=alpha; d.beta=beta;
    d.conv=p.conv; d.ssm_a=p.ssm_a; d.dt=p.dt; d.norm=p.norm;
    d.conv_state=p.conv_state; d.rec_state=p.rec_state; d.out=dn;
    delta_net(model,d);
    linear(model,*p.out_w,dn,p.out);
}
void Backend::linear_many(Gguf& model, std::span<const LinearRequest> requests, std::span<const float> x) {
    for (const auto& request : requests) linear(model, *request.tensor, x, request.output);
}
void Backend::linear_batch(Gguf& model, const Tensor& tensor, std::span<const float> x,
                           std::span<float> y, int K) {
    if (tensor.shape.size() != 2) throw std::invalid_argument("linear_batch expects a matrix");
    const size_t cols = static_cast<size_t>(tensor.shape[0]), rows = static_cast<size_t>(tensor.shape[1]);
    if (x.size() != cols * static_cast<size_t>(K) || y.size() != rows * static_cast<size_t>(K))
        throw std::invalid_argument("linear_batch shape mismatch");
    for (int k = 0; k < K; ++k)
        linear(model, tensor, x.subspan(static_cast<size_t>(k) * cols, cols),
               y.subspan(static_cast<size_t>(k) * rows, rows));
}
void Backend::feed_forward(Gguf& model, const Tensor& gate, const Tensor& up,
                           const Tensor& down, std::span<const float> x, std::span<float> y) {
    if(gate.shape.size()!=2 || up.shape!=gate.shape || down.shape.size()!=2 ||
       gate.shape[0]!=x.size() || down.shape[0]!=gate.shape[1] || down.shape[1]!=y.size())
        throw std::invalid_argument("Feed-forward shape mismatch");
    std::vector<float> g(static_cast<size_t>(gate.shape[1])), u(g.size());
    linear(model,gate,x,g);linear(model,up,x,u);
    for(size_t i=0;i<g.size();++i) {
        float e=std::exp(-std::abs(g[i]));
        float s=g[i]>=0?1.f/(1.f+e):e/(1.f+e);
        g[i]=g[i]*s*u[i];
    }
    linear(model,down,g,y);
}
namespace {
class Cpu final: public Backend {
public:
    std::string name() const override { return "CPU FP32 accumulation"; }
    void linear(Gguf& model, const Tensor& t, std::span<const float> x, std::span<float> y) override {
        if (t.shape.size() != 2 || t.shape[0] != x.size() || t.shape[1] != y.size())
            throw std::invalid_argument("Linear shape mismatch: " + t.name);
        const auto row_bytes = encoded_bytes(t.type, x.size());
        const auto rows = std::max<uint64_t>(1, (4 * 1024 * 1024) / row_bytes);
        std::vector<std::byte> buffer(static_cast<size_t>(std::min<uint64_t>(rows, y.size()) * row_bytes));
        for (uint64_t start = 0; start < y.size(); start += rows) {
            auto count = std::min<uint64_t>(rows, y.size() - start);
            auto tile = std::span(buffer).first(static_cast<size_t>(count * row_bytes));
            model.read(t, start * row_bytes, tile);
            matvec(t.type, tile, x, y.subspan(static_cast<size_t>(start), static_cast<size_t>(count)));
        }
    }
};
}
std::unique_ptr<Backend> cpu_backend() { return std::make_unique<Cpu>(); }
}
