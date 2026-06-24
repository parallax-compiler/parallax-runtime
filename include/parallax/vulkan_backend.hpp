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

// Device capabilities relevant to codegen and the memory model. Detected from the
// physical device during initialize(), before the logical device is created, so
// that features like bufferDeviceAddress can be enabled when supported. Consumed
// by the arena (buffer_device_address) and the compiler (int64/float64 gating).
struct DeviceCapabilities {
    bool shader_int64 = false;
    bool shader_float64 = false;
    bool buffer_device_address = false;
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
    const DeviceCapabilities& capabilities() const { return capabilities_; }

private:
    bool create_instance();
    bool select_physical_device();
    bool create_logical_device();
    void detect_capabilities();

    QueueFamilyIndices find_queue_families(VkPhysicalDevice device);
    bool is_device_suitable(VkPhysicalDevice device);
    
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    
    QueueFamilyIndices queue_indices_;
    VkPhysicalDeviceProperties device_properties_;
    DeviceCapabilities capabilities_;

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
