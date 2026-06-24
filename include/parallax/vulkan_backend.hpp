#ifndef PARALLAX_VULKAN_BACKEND_HPP
#define PARALLAX_VULKAN_BACKEND_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <string>

namespace parallax {

struct QueueFamilyIndices {
    std::optional<uint32_t> compute_family;
    
    bool is_complete() const {
        return compute_family.has_value();
    }
};

class VulkanBackend {
public:
    VulkanBackend();
    ~VulkanBackend();
    
    // Disable copy
    VulkanBackend(const VulkanBackend&) = delete;
    VulkanBackend& operator=(const VulkanBackend&) = delete;
    
    // Initialize Vulkan
    bool initialize();
    void cleanup();
    
    // Getters
    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physical_device() const { return physical_device_; }
    VkDevice device() const { return device_; }
    VkQueue compute_queue() const { return compute_queue_; }
    uint32_t compute_queue_family() const { return queue_indices_.compute_family.value(); }
    
    // Device info
    std::string device_name() const;
    uint32_t api_version() const;
    
private:
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    
    QueueFamilyIndices find_queue_families(VkPhysicalDevice device);
    bool is_device_suitable(VkPhysicalDevice device);
    
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    
    QueueFamilyIndices queue_indices_;
    VkPhysicalDeviceProperties device_properties_;
    
    // True only when the validation layer is actually present at runtime. Built
    // with PARALLAX_ENABLE_VALIDATION we *request* validation, but if the layer
    // is not installed we degrade gracefully instead of failing instance creation.
    bool validation_enabled_ = false;

    // These are declared unconditionally so that sizeof(VulkanBackend) is the
    // same in every translation unit. Guarding them with PARALLAX_ENABLE_VALIDATION
    // made the class layout depend on a private compile flag, so a TU that did not
    // see the flag allocated a smaller object than the constructor initialized —
    // an ODR violation and stack-buffer overflow. The *code* that uses them stays
    // guarded; only the data members must always be present.
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    bool setup_debug_messenger();
};

} // namespace parallax

#endif // PARALLAX_VULKAN_BACKEND_HPP
