#include "parallax/vulkan_backend.hpp"
#include <iostream>
#include <cstdlib>

int main() {
    std::cout << "Parallax Vulkan Backend Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    parallax::VulkanBackend backend;
    
    if (!backend.initialize()) {
        std::cerr << "\n❌ Vulkan backend unavailable" << std::endl;
        std::cerr << "\nThis is expected on systems without Vulkan support." << std::endl;
        std::cerr << "Parallax will automatically fall back to CPU execution." << std::endl;
        std::cerr << "\nTo enable GPU acceleration:" << std::endl;
        std::cerr << "  - Linux: Install vulkan-icd-loader and GPU drivers" << std::endl;
        std::cerr << "  - macOS: Install MoltenVK (brew install molten-vk)" << std::endl;
        std::cerr << "  - Windows: Install Vulkan SDK from vulkan.lunarg.com" << std::endl;
        return 0; // Not a failure - just no GPU
    }
    
    std::cout << "\n✓ Vulkan backend initialized successfully!" << std::endl;
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << backend.device_name() << std::endl;
    std::cout << "  API Version: " << VK_API_VERSION_MAJOR(backend.api_version()) 
              << "." << VK_API_VERSION_MINOR(backend.api_version())
              << "." << VK_API_VERSION_PATCH(backend.api_version()) << std::endl;
    std::cout << "  Compute Queue Family: " << backend.compute_queue_family() << std::endl;
    
    std::cout << "\nParallax will automatically use this GPU for std::execution::par" << std::endl;
    
    return 0;
}
