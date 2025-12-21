#!/bin/bash
set -e

SHADER_DIR="$(dirname "$0")"
OUTPUT_DIR="$SHADER_DIR/../include/parallax/shaders"

mkdir -p "$OUTPUT_DIR"

echo "Compiling shaders..."

# Compile vector_multiply.comp
glslangValidator -V "$SHADER_DIR/vector_multiply.comp" -o "$SHADER_DIR/vector_multiply.spv"

# Convert to C header
echo "Generating C++ header..."
cat > "$OUTPUT_DIR/vector_multiply.hpp" << 'EOF'
#ifndef PARALLAX_SHADER_VECTOR_MULTIPLY_HPP
#define PARALLAX_SHADER_VECTOR_MULTIPLY_HPP

#include <cstdint>

namespace parallax {
namespace shaders {

const uint32_t VECTOR_MULTIPLY_SPV[] = {
EOF

# Convert SPIR-V to C array
xxd -i < "$SHADER_DIR/vector_multiply.spv" | sed 's/unsigned/const uint32_t/g' | sed 's/char/uint8_t/g' >> "$OUTPUT_DIR/vector_multiply.hpp"

cat >> "$OUTPUT_DIR/vector_multiply.hpp" << 'EOF'
};

const size_t VECTOR_MULTIPLY_SPV_SIZE = sizeof(VECTOR_MULTIPLY_SPV);

} // namespace shaders
} // namespace parallax

#endif
EOF

echo "✓ Shader compiled and embedded"
