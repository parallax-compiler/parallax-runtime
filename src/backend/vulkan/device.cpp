#include "parallax/vulkan_backend.hpp"
#include <iostream>
#include <cstring>
#include <set>

namespace parallax {

VulkanBackend::VulkanBackend() = default;

VulkanBackend::~VulkanBackend() {
    cleanup();
}

bool VulkanBackend::initialize() {
    if (!create_instance()) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
#ifdef PARALLAX_ENABLE_VALIDATION
    if (!setup_debug_messenger()) {
        std::cerr << "Failed to setup debug messenger" << std::endl;
        return false;
    }
#endif
    
    if (!select_physical_device()) {
        std::cerr << "Failed to find suitable GPU" << std::endl;
        return false;
    }
    
    if (!create_logical_device()) {
        std::cerr << "Failed to create logical device" << std::endl;
        return false;
    }
    
    std::cout << "Parallax initialized on: " << device_name() << std::endl;
    return true;
}

void VulkanBackend::cleanup() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    
#ifdef PARALLAX_ENABLE_VALIDATION
    if (debug_messenger_ != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance_, debug_messenger_, nullptr);
        }
    }
#endif
    
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

bool VulkanBackend::create_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "Parallax Application";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "Parallax Runtime";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    // Use 1.2 for MoltenVK compatibility on macOS
    app_info.apiVersion = VK_API_VERSION_1_2;

    
    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    
    // Extensions for macOS MoltenVK support
    std::vector<const char*> extensions;
    
#ifdef __APPLE__
    // Required for MoltenVK
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back("VK_KHR_get_physical_device_properties2");
    create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
    
#ifdef PARALLAX_ENABLE_VALIDATION
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    
    const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = validation_layers;
#else
    create_info.enabledLayerCount = 0;
#endif
    
    create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    
    VkResult result = vkCreateInstance(&create_info, nullptr, &instance_);
    
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed with error code: " << result << std::endl;
        std::cerr << "Note: On macOS, ensure MoltenVK is installed (brew install molten-vk)" << std::endl;
    }
    
    return result == VK_SUCCESS;
}

bool VulkanBackend::select_physical_device() {
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    
    if (device_count == 0) {
        std::cerr << "No Vulkan-capable GPUs found" << std::endl;
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
    
    // Find first suitable device
    for (const auto& device : devices) {
        if (is_device_suitable(device)) {
            physical_device_ = device;
            queue_indices_ = find_queue_families(device);
            vkGetPhysicalDeviceProperties(physical_device_, &device_properties_);
            return true;
        }
    }
    
    std::cerr << "No suitable GPU found" << std::endl;
    return false;
}

bool VulkanBackend::is_device_suitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = find_queue_families(device);
    return indices.is_complete();
}

QueueFamilyIndices VulkanBackend::find_queue_families(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
    
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());
    
    int i = 0;
    for (const auto& queue_family : queue_families) {
        if (queue_family.queueFlags & VK_QUEUE_COMPUTE_BIT) {
            indices.compute_family = i;
            break;
        }
        i++;
    }
    
    return indices;
}

bool VulkanBackend::create_logical_device() {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_indices_.compute_family.value();
    queue_create_info.queueCount = 1;
    
    float queue_priority = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priority;
    
    VkPhysicalDeviceVulkan11Features vulkan11_features{};
    vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11_features.variablePointersStorageBuffer = VK_TRUE;
    vulkan11_features.variablePointers = VK_TRUE;

    VkPhysicalDeviceFeatures device_features{};
    
    // Extensions for MoltenVK
    std::vector<const char*> device_extensions;
#ifdef __APPLE__
    device_extensions.push_back("VK_KHR_portability_subset");
#endif
    
    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &vulkan11_features;
    create_info.pQueueCreateInfos = &queue_create_info;
    create_info.queueCreateInfoCount = 1;
    create_info.pEnabledFeatures = &device_features;
    create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();
    
#ifdef PARALLAX_ENABLE_VALIDATION
    const char* validation_layers[] = {"VK_LAYER_KHRONOS_validation"};
    create_info.enabledLayerCount = 1;
    create_info.ppEnabledLayerNames = validation_layers;
#else
    create_info.enabledLayerCount = 0;
#endif
    
    VkResult result = vkCreateDevice(physical_device_, &create_info, nullptr, &device_);
    if (result != VK_SUCCESS) {
        std::cerr << "vkCreateDevice failed with error code: " << result << std::endl;
        return false;
    }
    
    vkGetDeviceQueue(device_, queue_indices_.compute_family.value(), 0, &compute_queue_);
    return true;
}

std::string VulkanBackend::device_name() const {
    return std::string(device_properties_.deviceName);
}

uint32_t VulkanBackend::api_version() const {
    return device_properties_.apiVersion;
}

#ifdef PARALLAX_ENABLE_VALIDATION
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void* user_data) {
    
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::cerr << "Vulkan validation: " << callback_data->pMessage << std::endl;
    }
    return VK_FALSE;
}

bool VulkanBackend::setup_debug_messenger() {
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = 
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = 
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = debug_callback;
    
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
    
    if (func == nullptr) {
        return false;
    }
    
    return func(instance_, &create_info, nullptr, &debug_messenger_) == VK_SUCCESS;
}
#endif

} // namespace parallax
