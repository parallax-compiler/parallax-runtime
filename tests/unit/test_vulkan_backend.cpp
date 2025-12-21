#include "parallax/vulkan_backend.hpp"
#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "Parallax Vulkan Backend Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Check for MoltenVK ICD on macOS
#ifdef __APPLE__
    const char* vk_icd = std::getenv("VK_ICD_FILENAMES");
    if (!vk_icd) {
        std::cout << "\nNote: VK_ICD_FILENAMES not set. Setting to MoltenVK..." << std::endl;
        setenv("VK_ICD_FILENAMES", "/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json", 1);
    } else {
        std::cout << "\nVK_ICD_FILENAMES: " << vk_icd << std::endl;
    }
#endif
    
    parallax::VulkanBackend backend;
    
    if (!backend.initialize()) {
        std::cerr << "\n❌ Failed to initialize Vulkan backend" << std::endl;
        std::cerr << "\nThis is expected on systems without Vulkan support." << std::endl;
        std::cerr << "The runtime will fall back to CPU execution." << std::endl;
        std::cerr << "\nTo enable GPU acceleration on macOS:" << std::endl;
        std::cerr << "  1. Install MoltenVK: brew install molten-vk" << std::endl;
        std::cerr << "  2. Set environment: export VK_ICD_FILENAMES=/opt/homebrew/share/vulkan/icd.d/MoltenVK_icd.json" << std::endl;
        return 0; // Not a failure - just no GPU
    }
    
    std::cout << "\n✓ Vulkan backend initialized successfully!" << std::endl;
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << backend.device_name() << std::endl;
    std::cout << "  API Version: " << VK_API_VERSION_MAJOR(backend.api_version()) 
              << "." << VK_API_VERSION_MINOR(backend.api_version())
              << "." << VK_API_VERSION_PATCH(backend.api_version()) << std::endl;
    std::cout << "  Compute Queue Family: " << backend.compute_queue_family() << std::endl;
    
    return 0;
}
