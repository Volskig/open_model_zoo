// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace detection {

struct DetectedObject {
    cv::Rect rect;
    float confidence;

    explicit DetectedObject(const cv::Rect& rect = cv::Rect(), float confidence = -1.0f)
        : rect(rect), confidence(confidence) {}
};

using DetectedObjects = std::vector<DetectedObject>;

struct DetectorConfig {
    float confidence_threshold{0.6f};
    float increase_scale_x{1.15f};
    float increase_scale_y{1.15f};
};

class FaceDetection {
private:
    DetectorConfig config_;
    std::string input_name_;
    std::string output_name_;
    int max_detections_count_ = 0;
    int object_size_ = 0;
    int enqueued_frames_ = 0;
    float width_ = 0;
    float height_ = 0;

public:
    explicit FaceDetection(const DetectorConfig& config) : config_(config) {}

    DetectedObjects fetchResults(const cv::Mat&, const cv::Mat&);
};

}  // namespace detection
