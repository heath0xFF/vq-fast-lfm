# Source on the VQ after copying scripts/env.sh and this file to ~/lfm-fast/.
source "$HOME/lfm-fast/env.sh" || return 1
export L="$LFM/lcpp-master"
# Prefer this build over the GenieX libraries set by env.sh.
export LD_LIBRARY_PATH="$L/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$L/lib"
export D_Q8="${D_Q8:-$HOME/models/LFM2.5-2.6B-DSpark-Q8_0.gguf}"
