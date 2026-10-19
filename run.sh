#!/usr/bin/env bash
# Canonical launcher for Docullence.
#
# ARROW_DEFAULT_MEMORY_POOL=system is the important one: pyarrow (pulled in by
# Streamlit) bundles jemalloc, whose background thread segfaults intermittently
# ("jemalloc_bg_thd ... segfault in libarrow.so"), killing the server with no
# Python traceback - sometimes mid-upload. The system allocator avoids it.
# The OMP/tokenizer vars prevent torch/MuPDF OpenMP aborts.
set -e
cd "$(dirname "$0")"

# The real fix: Arrow's bundled jemalloc is built with the "je_arrow_" prefix, so
# it reads JE_ARROW_MALLOC_CONF (not MALLOC_CONF). Disabling its background thread
# stops the segfault at the source. ARROW_DEFAULT_MEMORY_POOL=system is a belt-and-
# suspenders fallback.
export JE_ARROW_MALLOC_CONF=background_thread:false
export ARROW_DEFAULT_MEMORY_POOL=system
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

PORT="${PORT:-8501}"
exec .venv/bin/streamlit run app.py --server.port "$PORT"
