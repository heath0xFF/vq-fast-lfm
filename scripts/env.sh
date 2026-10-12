# Source on the VQ: sets up geniex-bench env.
export LFM=$HOME/lfm-fast
export B=$LFM/bench/geniex-bench-linux-arm64-v0.8.0
export ADRENO=$HOME/.local/share/geniex-deps/root/usr/lib/aarch64-linux-gnu
mkdir -p $LFM/rpclib && ln -sf /usr/lib/aarch64-linux-gnu/libcdsprpc.so.1 $LFM/rpclib/libcdsprpc.so && ln -sf /usr/lib/aarch64-linux-gnu/libadsprpc.so.1 $LFM/rpclib/libadsprpc.so
export LD_LIBRARY_PATH="$LFM/rpclib:$B/lib:$B/lib/llama_cpp:$ADRENO${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OCL_ICD_FILENAMES=libOpenCL_adreno.so.1
export GENIEX_PLUGIN_PATH="$B/lib"
export BENCH="$B/bin/geniex-bench"
export M_USER=$HOME/models/lfm25-2.6b-q4_0/LFM2.5-2.6B-Q4_0.gguf
export M_GENIEX=$HOME/.cache/geniex/models/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q4_0.gguf
export M_QAD=$HOME/.cache/geniex/models/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf
