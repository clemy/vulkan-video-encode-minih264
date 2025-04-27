#include "videoencoder.hpp"

#include <array>
#include <cassert>
#include <cstring>

#include "utility.hpp"
#define H264E_SVC_API 0
#define H264E_MAX_THREADS 0
#define H264E_ENABLE_DENOISE 0
#define MAX_LONG_TERM_FRAMES 1
#include "minih264e.h"

void VideoEncoder::init(VkPhysicalDevice physicalDevice, VkDevice device, VmaAllocator allocator,
                        uint32_t computeQueueFamily, VkQueue computeQueue, VkCommandPool computeCommandPool,
                        const std::vector<VkImage>& inputImages,
                        const std::vector<VkImageView>& inputImageViews, uint32_t width, uint32_t height,
                        uint32_t fps) {
    assert(!m_running);

    if (m_initialized) {
        if ((width & ~1) == m_width && (height & ~1) == m_height) {
            // nothing changed
            return;
        }

        // resolution changed
        deinit();
    }

    m_physicalDevice = physicalDevice;
    m_device = device;
    m_allocator = allocator;
    m_computeQueue = computeQueue;
    m_computeQueueFamily = computeQueueFamily;
    m_computeCommandPool = computeCommandPool;
    m_inputImages = inputImages;
    m_width = width & ~1;
    m_height = height & ~1;

    allocateIntermediateImage();
    allocateTransferBuffer();
    createYCbCrConversionPipeline(inputImageViews);
    initializeCodec();

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(m_device, &fenceInfo, nullptr, &m_encodeFinishedFence));

    m_frameCount = 0;
    m_initialized = true;
}

void VideoEncoder::queueEncode(uint32_t currentImageIx) {
    assert(!m_running);
    convertRGBtoYCbCr(currentImageIx);
    m_running = true;
}

void VideoEncoder::finishEncode(const char*& data, size_t& size) {
    if (!m_running) {
        size = 0;
        return;
    }

    getOutputVideoPacket(data, size);

    vkFreeCommandBuffers(m_device, m_computeCommandPool, 1, &m_computeCommandBuffer);
    m_frameCount++;

    m_running = false;
}

void VideoEncoder::allocateIntermediateImage() {
    VkImageCreateInfo tmpImgCreateInfo;
    tmpImgCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    tmpImgCreateInfo.pNext = nullptr;
    tmpImgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    tmpImgCreateInfo.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    tmpImgCreateInfo.extent = {m_width, m_height, 1};
    tmpImgCreateInfo.mipLevels = 1;
    tmpImgCreateInfo.arrayLayers = 1;
    tmpImgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    tmpImgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    tmpImgCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    tmpImgCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    tmpImgCreateInfo.queueFamilyIndexCount = 0;
    tmpImgCreateInfo.pQueueFamilyIndices = nullptr;
    tmpImgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // this is causing a validation error as Nvidia driver is not returning those create flags
    // in vkGetPhysicalDeviceVideoFormatPropertiesKHR
    tmpImgCreateInfo.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VK_CHECK(
        vmaCreateImage(m_allocator, &tmpImgCreateInfo, &allocInfo, &m_yCbCrImage, &m_yCbCrImageAllocation, nullptr));

    VkImageViewUsageCreateInfo viewUsageInfo = {};
    viewUsageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
    viewUsageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.pNext = &viewUsageInfo;
    viewInfo.image = m_yCbCrImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_yCbCrImageView));

    viewUsageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    uint32_t numPlanes = 3;
    m_yCbCrImagePlaneViews.resize(numPlanes);
    viewInfo.format = VK_FORMAT_R8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT;
    VK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_yCbCrImagePlaneViews[0]));
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_1_BIT;
    VK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_yCbCrImagePlaneViews[1]));
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_PLANE_2_BIT;
    VK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_yCbCrImagePlaneViews[2]));
}

void VideoEncoder::allocateTransferBuffer() {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = m_width * m_height * 3 / 2;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.pNext = nullptr;
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
    VK_CHECK(vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &m_transferBuffer, &m_transferBufferAllocation,
                                nullptr));

    VK_CHECK(vmaMapMemory(m_allocator, m_transferBufferAllocation, reinterpret_cast<void**>(&m_transferData)));
}

void VideoEncoder::createYCbCrConversionPipeline(const std::vector<VkImageView>& inputImageViews) {
    const char* shaderFileName = "shaders/rgb-ycbcr-shader-3plane.comp.spv";
    printf("Using %s\n", shaderFileName);
    auto computeShaderCode = readFile(shaderFileName);
    VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                        .codeSize = computeShaderCode.size(),
                                        .pCode = reinterpret_cast<const uint32_t*>(computeShaderCode.data())};
    VkShaderModule computeShaderModule;
    VK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &computeShaderModule));
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    std::array<VkDescriptorSetLayoutBinding, 4> layoutBindings{};
    for (uint32_t i = 0; i < layoutBindings.size(); i++) {
        layoutBindings[i].binding = i;
        layoutBindings[i].descriptorCount = 1;
        layoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        layoutBindings[i].pImmutableSamplers = nullptr;
        layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1 + m_yCbCrImagePlaneViews.size();
    layoutInfo.pBindings = layoutBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_computeDescriptorSetLayout));

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_computeDescriptorSetLayout;
    VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_computePipelineLayout));

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = m_computePipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;
    VK_CHECK(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_computePipeline));

    vkDestroyShaderModule(m_device, computeShaderModule, nullptr);

    const int maxFramesCount = static_cast<uint32_t>(inputImageViews.size());
    std::array<VkDescriptorPoolSize, 1> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 4 * maxFramesCount;
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxFramesCount;
    VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));

    std::vector<VkDescriptorSetLayout> layouts(maxFramesCount, m_computeDescriptorSetLayout);
    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = m_descriptorPool;
    descAllocInfo.descriptorSetCount = maxFramesCount;
    descAllocInfo.pSetLayouts = layouts.data();

    m_computeDescriptorSets.resize(maxFramesCount);
    VK_CHECK(vkAllocateDescriptorSets(m_device, &descAllocInfo, m_computeDescriptorSets.data()));
    for (size_t i = 0; i < maxFramesCount; i++) {
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
        std::array<VkDescriptorImageInfo, 4> imageInfos{};

        imageInfos[0].imageView = inputImageViews[i];
        imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos[0].sampler = VK_NULL_HANDLE;
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = m_computeDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pImageInfo = &imageInfos[0];

        for (uint32_t p = 0; p < m_yCbCrImagePlaneViews.size(); ++p) {
            imageInfos[p + 1].imageView = m_yCbCrImagePlaneViews[p];
            imageInfos[p + 1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageInfos[p + 1].sampler = VK_NULL_HANDLE;
            descriptorWrites[p + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[p + 1].dstSet = m_computeDescriptorSets[i];
            descriptorWrites[p + 1].dstBinding = p + 1;
            descriptorWrites[p + 1].dstArrayElement = 0;
            descriptorWrites[p + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[p + 1].descriptorCount = 1;
            descriptorWrites[p + 1].pImageInfo = &imageInfos[p + 1];
        }

        vkUpdateDescriptorSets(m_device, 1 + m_yCbCrImagePlaneViews.size(), descriptorWrites.data(), 0, nullptr);
    }
}

void VideoEncoder::initializeCodec() {
    H264E_create_param_t createParams{
        .width = (int)m_width,
        .height = (int)m_height,
        .gop = 16,
        .vbv_size_bytes = 0,
        .vbv_overflow_empty_frame_flag = 0,
        .vbv_underflow_stuffing_flag = 0,
        .fine_rate_control_flag = 0,
        .const_input_flag = 0,
        .max_long_term_reference_frames = 0,
        .enableNEON = 0,
        .temporal_denoise_flag = 0,
        .sps_id = 0
    };
    int sizeof_persist, sizeof_scratch;
    int err = H264E_sizeof(&createParams, &sizeof_persist, &sizeof_scratch);
    if (err != 0) {
        throw std::runtime_error("Error: H264E_sizeof returned " + std::to_string(err));
    }
    m_codecDataPersist = new char[sizeof_persist];
    m_codecDataScratch = new char[sizeof_scratch];
    err = H264E_init((H264E_persist_t*)m_codecDataPersist, &createParams);
    if (err != 0) {
        throw std::runtime_error("Error: H264E_persist_t returned " + std::to_string(err));
    }
}

void VideoEncoder::convertRGBtoYCbCr(uint32_t currentImageIx) {
    // begin command buffer for compute shader
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = m_computeCommandPool;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &m_computeCommandBuffer));
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(m_computeCommandBuffer, &beginInfo));

    std::vector<VkImageMemoryBarrier2> barriers;
    VkImageMemoryBarrier2 imageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }};
    // transition YCbCr image (luma and chroma planes) to be shader target
    imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_NONE;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarrier.image = m_yCbCrImage;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    if (m_yCbCrImagePlaneViews.size() >= 3)
        imageMemoryBarrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_PLANE_2_BIT;
    barriers.push_back(imageMemoryBarrier);
    // transition source image to be shader source
    imageMemoryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    imageMemoryBarrier.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageMemoryBarrier.image = m_inputImages[currentImageIx];
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers.push_back(imageMemoryBarrier);
    VkDependencyInfoKHR dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
                                       .imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size()),
                                       .pImageMemoryBarriers = barriers.data()};
    vkCmdPipelineBarrier2(m_computeCommandBuffer, &dependencyInfo);

    // run the RGB->YCbCr conversion shader
    vkCmdBindPipeline(m_computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    vkCmdBindDescriptorSets(m_computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0, 1,
                            &m_computeDescriptorSets[currentImageIx], 0, 0);
    vkCmdDispatch(m_computeCommandBuffer, (m_width + 15) / 16, (m_height + 15) / 16,
                  1);  // work item local size = 16x16

    // transition source image to be transfer source
    VkImageMemoryBarrier2 imageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .image = m_yCbCrImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT | VK_IMAGE_ASPECT_PLANE_2_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }};
    VkDependencyInfoKHR dependencyInfo2{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
                                       .imageMemoryBarrierCount = 1,
                                       .pImageMemoryBarriers = &imageMemoryBarrier2};
    vkCmdPipelineBarrier2(m_computeCommandBuffer, &dependencyInfo2);

    std::vector<VkBufferImageCopy> imageCopies;
    VkDeviceSize offset = 0;
    uint32_t width = m_width;
    uint32_t height = m_height;
    for (int p = 0; p < 3; p++) {
        VkImageAspectFlags aspect;
        switch (p)
        {
        case 0:
            aspect = VK_IMAGE_ASPECT_PLANE_0_BIT;
            break;
        case 1:
            aspect = VK_IMAGE_ASPECT_PLANE_1_BIT;
            break;
        case 2:
            aspect = VK_IMAGE_ASPECT_PLANE_2_BIT;
            break;
        default:
            throw std::runtime_error("Unknown aspect");
        }
        VkBufferImageCopy imageCopy{
            .bufferOffset = offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = aspect,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1}
        };
        imageCopies.push_back(imageCopy);
        offset += width * height;
        if (p == 0) {
            width /= 2;
            height /= 2;
        }
    }
    vkCmdCopyImageToBuffer(m_computeCommandBuffer, m_yCbCrImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_transferBuffer, imageCopies.size(), imageCopies.data());

    VK_CHECK(vkEndCommandBuffer(m_computeCommandBuffer));
    VkSubmitInfo submitInfo{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1,
                            .pCommandBuffers = &m_computeCommandBuffer};
    VK_CHECK(vkResetFences(m_device, 1, &m_encodeFinishedFence));
    VK_CHECK(vkQueueSubmit(m_computeQueue, 1, &submitInfo, m_encodeFinishedFence));
}

void VideoEncoder::getOutputVideoPacket(const char*& data, size_t& size) {
    VK_CHECK(vkWaitForFences(m_device, 1, &m_encodeFinishedFence, VK_TRUE, std::numeric_limits<uint64_t>::max()));

    H264E_run_param_t runParams{
        .encode_speed = H264E_SPEED_SLOWEST,
        .frame_type = H264E_FRAME_TYPE_DEFAULT,
        .long_term_idx_use = 0,
        .long_term_idx_update = 0,
        .desired_frame_bytes = 0,
        .qp_min = 26,
        .qp_max = 26,
        .desired_nalu_bytes = 0,
        .nalu_callback = nullptr,
        .nalu_callback_token = nullptr
    };
    uint32_t imgSize = m_width * m_height;
    H264E_io_yuv_t yuv{
        .yuv = {m_transferData, m_transferData + imgSize, m_transferData + imgSize + imgSize / 4},
        .stride = {(int)m_width, (int)m_width / 2, (int)m_width / 2}
    };
    
    int codedSize;
    int err = H264E_encode((H264E_persist_t *)m_codecDataPersist, (H264E_scratch_t *)m_codecDataScratch,
                           &runParams, &yuv,
                           (unsigned char**)&data, &codedSize);
    if (err != 0) {
        throw std::runtime_error("Error: H264E_encode returned " + std::to_string(err));
    }
    size = codedSize;
    // printf("Encoded frame %d, status %d, offset %d, size %zd\n", m_frameCount, err, 0, size);
 }

void VideoEncoder::deinit() {
    if (!m_initialized) {
        return;
    }

    if (m_running) {
        const char* data;
        size_t size;
        getOutputVideoPacket(data, size);
        vkFreeCommandBuffers(m_device, m_computeCommandPool, 1, &m_computeCommandBuffer);
    }
    vkDestroyFence(m_device, m_encodeFinishedFence, nullptr);
    vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_computePipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_computeDescriptorSetLayout, nullptr);

    for (uint32_t i = 0; i < m_yCbCrImagePlaneViews.size(); i++) {
        vkDestroyImageView(m_device, m_yCbCrImagePlaneViews[i], nullptr);
    }
    m_yCbCrImagePlaneViews.clear();
    vkDestroyImageView(m_device, m_yCbCrImageView, nullptr);
    vmaDestroyImage(m_allocator, m_yCbCrImage, m_yCbCrImageAllocation);

    vmaUnmapMemory(m_allocator, m_transferBufferAllocation);
    vmaDestroyBuffer(m_allocator, m_transferBuffer, m_transferBufferAllocation);

    delete [] m_codecDataPersist;
    delete [] m_codecDataScratch;

    m_initialized = false;
}
