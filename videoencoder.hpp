#pragma once
#define VK_NO_PROTOTYPES
#ifdef __linux__
    #include <vk_mem_alloc.h>
    #include <volk.h>
#else
    #include <vma/vk_mem_alloc.h>
    #include <volk/volk.h>
#endif

#include <vector>

class VideoEncoder {
   public:
    void init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator, uint32_t computeQueueFamily,
              VkQueue computeQueue, VkCommandPool computeCommandPool,
              const std::vector<VkImage>& inputImages, const std::vector<VkImageView>& inputImageViews, uint32_t width,
              uint32_t height, uint32_t fps);
    void queueEncode(uint32_t currentImageIx);
    void finishEncode(const char*& data, size_t& size);
    void deinit();

    ~VideoEncoder() { deinit(); }

   private:
    void allocateIntermediateImage();
    void allocateTransferBuffer();
    void createYCbCrConversionPipeline(const std::vector<VkImageView>& inputImageViews);
    void initializeCodec();

    void convertRGBtoYCbCr(uint32_t currentImageIx);
    void encodeVideoFrame();
    void getOutputVideoPacket(const char*& data, size_t& size);

    bool m_initialized{false};
    bool m_running{false};
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VmaAllocator m_allocator;
    VkQueue m_computeQueue;
    uint32_t m_computeQueueFamily;
    VkCommandPool m_computeCommandPool;
    std::vector<VkImage> m_inputImages;
    uint32_t m_width;
    uint32_t m_height;

    VkDescriptorSetLayout m_computeDescriptorSetLayout;
    VkPipelineLayout m_computePipelineLayout;
    VkPipeline m_computePipeline;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_computeDescriptorSets;

    VkImage m_yCbCrImage;
    VmaAllocation m_yCbCrImageAllocation;
    VkImageView m_yCbCrImageView;
    std::vector<VkImageView> m_yCbCrImagePlaneViews;

    VkBuffer m_transferBuffer;
    VmaAllocation m_transferBufferAllocation;
    unsigned char* m_transferData;

    void* m_codecDataPersist;
    void* m_codecDataScratch;

    uint32_t m_frameCount;

    VkFence m_encodeFinishedFence;
    VkCommandBuffer m_computeCommandBuffer;
};
