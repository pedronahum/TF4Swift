#!/usr/bin/env bash
# Auto-detect Homebrew libtensorflow paths on macOS (Apple Silicon or Intel)
BREW_PREFIX="$(brew --prefix libtensorflow)"

export LIBTENSORFLOW_INCLUDEDIR="$BREW_PREFIX/include"
export LIBTENSORFLOW_LIBDIR="$BREW_PREFIX/lib"

echo "Using:"
echo "  include: $LIBTENSORFLOW_INCLUDEDIR"
echo "  lib    : $LIBTENSORFLOW_LIBDIR"

# Build (adds rpath so you usually don't need DYLD_LIBRARY_PATH)
swift build \
  -Xcc -I"$LIBTENSORFLOW_INCLUDEDIR" \
  -Xlinker -L"$LIBTENSORFLOW_LIBDIR" \
  -Xlinker -rpath -Xlinker "$LIBTENSORFLOW_LIBDIR"

# Tests
swift test \
  -Xcc -I"$LIBTENSORFLOW_INCLUDEDIR" \
  -Xlinker -L"$LIBTENSORFLOW_LIBDIR" \
  -Xlinker -rpath -Xlinker "$LIBTENSORFLOW_LIBDIR"
