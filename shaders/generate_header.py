#!/usr/bin/env python3
import struct
import sys

def generate_spirv_header(spv_file, output_file):
    with open(spv_file, 'rb') as f:
        data = f.read()
    
    # Read as uint32 array (little-endian)
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    
    with open(output_file, 'w') as f:
        f.write('#ifndef PARALLAX_SHADERS_VECTOR_MULTIPLY_HPP\n')
        f.write('#define PARALLAX_SHADERS_VECTOR_MULTIPLY_HPP\n\n')
        f.write('#include <cstdint>\n')
        f.write('#include <cstddef>\n\n')
        f.write('namespace parallax {\n')
        f.write('namespace shaders {\n\n')
        f.write('alignas(4) static const uint32_t VECTOR_MULTIPLY_SPV[] = {\n')
        
        for i in range(0, len(words), 6):
            chunk = words[i:i+6]
            f.write('    ' + ', '.join(f'0x{w:08x}' for w in chunk) + ',\n')
        
        f.write('};\n\n')
        f.write(f'static const size_t VECTOR_MULTIPLY_SPV_SIZE = sizeof(VECTOR_MULTIPLY_SPV);\n\n')
        f.write('} // namespace shaders\n')
        f.write('} // namespace parallax\n\n')
        f.write('#endif // PARALLAX_SHADERS_VECTOR_MULTIPLY_HPP\n')

if __name__ == '__main__':
    generate_spirv_header('vector_multiply.spv', '../include/parallax/shaders/vector_multiply.hpp')
    print('Generated vector_multiply.hpp')
