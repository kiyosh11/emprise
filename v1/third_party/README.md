# Supporting dependencies

- `json.hpp`: nlohmann/json 3.12.0, MIT; see `json-LICENSE.MIT`.
- `pcre2-10.45/`: PCRE2 10.45, BSD-style license; see its `LICENCE` file.
- `utf8proc/`: utf8proc 2.10.0, MIT and Unicode data terms; see `LICENSE.md`.

These libraries provide JSON parsing, Unicode regex matching, and UTF-8 helpers.
They do not execute the language model. Matrix kernels, model execution, and
storage code in `src/` are the custom engine.

The downloaded llama.cpp binaries under ignored `.tools/` are a separate baseline,
MIT-licensed by their authors. NVIDIA runtime binaries in `.tools/` retain NVIDIA's
license terms. They are not redistributed as project source dependencies.

Algorithm/format references: ggml's Q6_K block layout and the pinned llama.cpp
`b10934` Qwen3.5 GGUF conversion/model implementation, plus Transformers' Qwen3.5
implementation. GGUF recurrent V heads use tiled order; the converter's RMS norm
weights and SSM decay parameters are already transformed from the HF originals.
