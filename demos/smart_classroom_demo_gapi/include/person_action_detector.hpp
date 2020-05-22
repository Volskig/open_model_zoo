// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <vector>
#include <numeric>
#include <opencv2/imgproc/imgproc.hpp>

struct Detections {
    cv::Rect rect;
    int action_label;
    float detection_conf;
    float action_conf; 

    Detections(const cv::Rect rect,
               const int action_label,
               const float detection_conf,
               const float action_conf) : rect(rect),
                                          action_label(action_label),
                                          detection_conf(detection_conf),
                                          action_conf(action_conf) {}
};

struct ActionDetectorConfig
{
    float nms_sigma = 0.6f;
    /** @brief Threshold for detected objects */
    float detection_confidence_threshold = 0.4f;
    /** @brief Threshold for recognized actions */
    float action_confidence_threshold = 0.75f;
    /** @brief Scale of action logits for the old network version */
    float old_action_scale = 3.f;
    /** @brief Scale of action logits for the new network version */
    float new_action_scale = 16.f;
    /** @brief Default action class label */
    int default_action_id = 0;
    /** @brief Number of top-score bboxes in output */
    int keep_top_k = 200;
    /** @brief Number of SSD anchors for the old network version */
    std::vector<int> old_anchors{4};
    /** @brief Number of SSD anchors for the new network version */
    std::vector<int> new_anchors{1, 4};
    /** @brief Number of actions to detect */
    size_t num_action_classes = 3;
    /** @brief  SSD bbox encoding variances */
    float variances[4]{0.1f, 0.1f, 0.2f, 0.2f};
};

class ActionDetector {
public:
    void GetPostProcResult(const float* local_data,
                           const float* det_conf_data,
                           const float* prior_data,
                           const std::vector<float*>& action_conf_data,
                           const cv::Size& frame_size,
                           std::vector<Detections>& out_detections);
private:
    ActionDetectorConfig config_;
    std::vector<int> head_ranges_;
    std::vector<int> head_step_sizes_;
    std::vector<std::vector<int>> glob_anchor_map_;
    int num_glob_anchors_;

    struct Bbox {
        float xmin;
        float ymin;
        float xmax;
        float ymax;
    };

    inline cv::Rect ConvertToRect(const Bbox& prior_bbox, const Bbox& variances,
                                  const Bbox& encoded_bbox, const cv::Size& frame_size) const;
    inline void SoftNonMaxSuppression(const std::vector<Detections>& detections,
                                      const float& sigma, const int& top_k, 
                                      const float& min_det_conf,
                                      std::vector<Detections>& out_detections) const;
    inline Bbox ParseBBoxRecord(const float* data) const; 
};
