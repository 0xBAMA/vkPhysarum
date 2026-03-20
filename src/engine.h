#pragma once

#include <iostream>
#include <chrono>
#include <thread>

#include <vk_types.h>
#include <vk_descriptors.h>
#include <vk_pipelines.h>
#include <vk_loader.h>

struct DeletionQueue {
	std::deque< std::function< void() > > deletors;

	// called when we add new Vulkan objects
	void push_function( std::function< void() >&& function ) {
		deletors.push_back( function );
	}

	// called during Cleanup()
	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for ( auto it = deletors.rbegin(); it != deletors.rend(); it++ ) {
			( *it )(); //call functors
		}
		deletors.clear();
	}
};

struct frameData_t {
	// frame sync primitives
	VkSemaphore swapchainSemaphore;
	VkFence renderFence;

	// command buffer + allocator
	VkCommandPool commandPool;
	VkCommandBuffer mainCommandBuffer;

	// handling frame-local resources
	DeletionQueue deletionQueue;

	// descriptor pool management
	DescriptorAllocatorGrowable frameDescriptors;
};

// common configuration across all shaders
struct GlobalData {
	glm::mat4 placeholder0;
	glm::mat4 placeholder1;
	glm::mat4 placeholder2;
	glm::vec4 placeholder3;
	glm::vec4 placeholder4;
	glm::vec4 placeholder5;
	glm::vec4 placeholder6;
};

struct PushConstants {
	glm::uvec2 floatBufferResolution;
	glm::uvec2 presentBufferResolution;
	uint32_t wangSeed;
	uint32_t operation;
};

constexpr unsigned int FRAME_OVERLAP = 2;
constexpr bool useValidationLayers = true;

// then the SSBO for the agents
struct Agent {
	float mass;
	float pad;
	float drag;
	float senseDistance;
	float senseAngle;
	float turnAngle;
	float forceAmount; // replaces stepsize
	float depositAmount;
	glm::vec2 position;
	glm::vec2 velocity;
};

class PrometheusInstance {
public:

// physarum data/storage resources
	uint32_t numAgents = 10000;
	AllocatedBuffer simAgentBuffer;

	glm::uvec2 FloatBufferResolution{ 1024, 512 };
	AllocatedImage FloatBufferA;
	AllocatedImage FloatBufferB;

// descriptor set layout for these resources:
	VkDescriptorSetLayout physarumGlobalDescriptorSetLayout;
	VkDescriptorSet physarumGlobalDescriptorSet;

// the stuff defining the push constants:
	PushConstants physarumGlobalPushConstant;

// and the pipeline layout which contains the information for the descriptors + push constants
	VkPipelineLayout physarumGlobalPipelineLayout;

// pipelines for physarum
	// agent update
	VkPipeline agentUpdatePipeline;

	// agent raster
	VkPipeline agentRasterPipeline;

	// buffer blur
	VkPipeline bufferBlurPipeline;

	// buffer raster
	VkPipeline bufferPresentPipeline;

	bool resizeRequest { false };
	bool isInitialized { false };
	bool stopRendering { false };
	int frameNumber { 0 };

	void initDefaultData ();
	// for buffer setup
	AllocatedBuffer createBuffer( size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage );
	void destroyBuffer( const AllocatedBuffer& buffer );

	// basic Vulkan necessities, environmental handles
	VkInstance instance;						// Vulkan library handle
	VkDebugUtilsMessengerEXT debugMessenger;	// debug output messenger
	VkPhysicalDevice physicalDevice;			// GPU handle for the physical device in use
	VkDevice device;							// the abstract device that we interact with
	VkSurfaceKHR surface;						// the Vulkan window surface

	// an image to draw into and eventually pass to the swapchain
	AllocatedImage drawImage;
	AllocatedImage depthImage;
	VkExtent2D drawExtent;
	float renderScale = 1.0f;

	// some helper functions for allocating textures
	AllocatedImage createImage ( VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false ); // storage image type
	AllocatedImage createImage ( void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false ); // loaded from disk
	void destroyImage ( const AllocatedImage& img );

	// and some default textures
	AllocatedImage whiteImage;
	AllocatedImage blackImage;
	AllocatedImage greyImage;

	// and default sampler types
	VkSampler defaultSamplerLinear;
	VkSampler defaultSamplerNearest;

	// our frameData struct, which contains command pool/buffer + sync primitive handles
	frameData_t frameData[ FRAME_OVERLAP ];
	frameData_t& getCurrentFrame () { return frameData[ frameNumber % FRAME_OVERLAP ]; }

	VkFence immediateFence;
	VkCommandBuffer immediateCommandBuffer;
	VkCommandPool immediateCommandPool;
	void immediateSubmit( std::function< void( VkCommandBuffer cmd ) > && function );

	GlobalData globalData;
	DescriptorAllocatorGrowable globalDescriptorAllocator;

	VkDescriptorSet drawImageDescriptors;
	VkDescriptorSetLayout drawImageDescriptorLayout;

	// the queue that we submit work to
	VkQueue graphicsQueue;
	uint32_t graphicsQueueFamilyIndex;

	// window size, swapchain size
	VkExtent2D windowExtent { 0,0 };
	VkExtent2D swapchainExtent;

	// swapchain handles
	VkSwapchainKHR swapchain;
	VkFormat swapchainImageFormat;
	std::vector< VkImage > swapchainImages;
	std::vector< VkImageView > swapchainImageViews;
	std::vector< VkSemaphore > swapchainPresentSemaphores;

	// handle for the AMD Vulkan Memory Allocator
	VmaAllocator allocator;

	// deletion queue automatically managing global resources
	DeletionQueue mainDeletionQueue;

	struct SDL_Window* window{ nullptr };
	static PrometheusInstance& Get ();

	void Init ();
	void Draw ();
	void MainLoop ();
	void ShutDown ();

private:
	// init helpers
	void initVulkan ();
	void initSwapchain ();
	void initCommandStructures ();
	void initSyncStructures ();
	void initDescriptors ();
	void initPipelines ();
	void initImgui ();
	void initResources ();

	// main loop helpers
	void drawImgui ( VkCommandBuffer cmd, VkImageView targetImageView );
	void drawGeometry ( VkCommandBuffer cmd );

	// swapchain helpers
	void resizeSwapchain ();
	void createSwapchain ( uint32_t w, uint32_t h );
	void destroySwapchain ();
};