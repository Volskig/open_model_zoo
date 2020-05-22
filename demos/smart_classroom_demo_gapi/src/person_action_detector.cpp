// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "person_action_detector.hpp"


#define SSD_LOCATION_RECORD_SIZE 4
#define SSD_PRIORBOX_RECORD_SIZE 4
#define NUM_DETECTION_CLASSES 2
#define POSITIVE_DETECTION_IDX 1
#define NUM_CANDIDATES 4300
#define INVALID_TOP_K_IDX -1

void ActionDetector::GetPostProcResult(const float* local_data,
                                       const float* det_conf_data,
                                       const float* prior_data,
                                       const std::vector<float*>& action_conf_data,
                                       const cv::Size& frame_size,
                                       std::vector<Detections>& out_detections) {
    std::vector<Detections> all_detections;
    
    const auto& head_anchors = config_.old_anchors;
    const int num_heads = head_anchors.size();

    head_ranges_.resize(num_heads + 1);
    glob_anchor_map_.resize(num_heads);
    head_step_sizes_.resize(num_heads);

    num_glob_anchors_ = 0;
    head_ranges_[0] = 0;        
    glob_anchor_map_[0].resize(head_anchors[0/*head_id*/]);
    for (int anchor_id = 0; anchor_id < head_anchors[0]; ++anchor_id) {
        glob_anchor_map_[0][anchor_id] = num_glob_anchors_++;
        head_step_sizes_[0] = 1;
    }
    for (int candidate = 0; candidate < NUM_CANDIDATES; ++candidate) {
        const float detection_conf = det_conf_data[candidate * NUM_DETECTION_CLASSES + POSITIVE_DETECTION_IDX];
        if (detection_conf < config_.detection_confidence_threshold) {
            continue;    
        }
        int action_label = -1;
        float action_conf = 0.f;
        int head_id = 0;
        const int head_p = candidate - head_ranges_[head_id];
                
        const int head_num_anchors = config_.old_anchors[head_id];
        const int anchor_id = head_p % head_num_anchors;
        const int glob_anchor_id = glob_anchor_map_[head_id][anchor_id];
        const float* anchor_conf_data = action_conf_data[glob_anchor_id];
        const int action_conf_idx_shift = head_p / head_num_anchors * config_.num_action_classes;
                
        const int action_conf_step = head_step_sizes_[head_id];
        const float scale = config_.old_action_scale;
        float action_max_exp_value = 0.f;
        float action_sum_exp_values = 0.f;
        for (size_t c = 0; c < config_.num_action_classes; ++c) {
            float action_exp_value = std::exp(scale * anchor_conf_data[action_conf_idx_shift + c * action_conf_step]);
            action_sum_exp_values += action_exp_value;
            if (action_exp_value > action_max_exp_value) {
                action_max_exp_value = action_exp_value;
                action_label = c;
            }
        }
        action_conf = action_max_exp_value / action_sum_exp_values;

        if (action_label < 0 || action_conf < config_.action_confidence_threshold) {
            action_label = config_.default_action_id;
            action_conf = 0.f;
        }  
        const auto priorbox = ParseBBoxRecord(prior_data + candidate * SSD_PRIORBOX_RECORD_SIZE);
        const auto encoded_box = ParseBBoxRecord(local_data + candidate * SSD_LOCATION_RECORD_SIZE);
        const auto variances   = ParseBBoxRecord(prior_data + (NUM_CANDIDATES + candidate) * SSD_PRIORBOX_RECORD_SIZE);
 
        all_detections.emplace_back(ConvertToRect(priorbox, variances, encoded_box, frame_size),
                                            action_label, detection_conf, action_conf);
        }
        SoftNonMaxSuppression(all_detections, 
                              config_.nms_sigma,
                              config_.keep_top_k,
                              config_.detection_confidence_threshold,
                              out_detections);
}

inline cv::Rect ActionDetector::ConvertToRect(const Bbox& prior_bbox,
                                              const Bbox& variances,
                                              const Bbox& encoded_bbox,
                                              const cv::Size& frame_size) const {
    const float prior_width = prior_bbox.xmax - prior_bbox.xmin;
    const float prior_height = prior_bbox.ymax - prior_bbox.ymin;
    const float prior_center_x = 0.5f * (prior_bbox.xmin + prior_bbox.xmax);
    const float prior_center_y = 0.5f * (prior_bbox.ymin + prior_bbox.ymax);

    /** Decode bbox coordinates from the SSD format **/
    const float decoded_bbox_center_x =
            variances.xmin * encoded_bbox.xmin * prior_width + prior_center_x;
    const float decoded_bbox_center_y =
            variances.ymin * encoded_bbox.ymin * prior_height + prior_center_y;
    const float decoded_bbox_width =
            static_cast<float>(exp(static_cast<float>(variances.xmax * encoded_bbox.xmax))) * prior_width;
    const float decoded_bbox_height =
            static_cast<float>(exp(static_cast<float>(variances.ymax * encoded_bbox.ymax))) * prior_height;

    /** Create decoded bbox **/
    const float decoded_bbox_xmin = decoded_bbox_center_x - 0.5f * decoded_bbox_width;
    const float decoded_bbox_ymin = decoded_bbox_center_y - 0.5f * decoded_bbox_height;
    const float decoded_bbox_xmax = decoded_bbox_center_x + 0.5f * decoded_bbox_width;
    const float decoded_bbox_ymax = decoded_bbox_center_y + 0.5f * decoded_bbox_height;

    /** Convert decoded bbox to CV_Rect **/
    return cv::Rect(static_cast<int>(decoded_bbox_xmin * frame_size.width),
                    static_cast<int>(decoded_bbox_ymin * frame_size.height),
                    static_cast<int>((decoded_bbox_xmax - decoded_bbox_xmin) * frame_size.width),
                    static_cast<int>((decoded_bbox_ymax - decoded_bbox_ymin) * frame_size.height));
}

inline void ActionDetector::SoftNonMaxSuppression(const std::vector<Detections>& detections,
                                                  const float& sigma,
                                                  const int& top_k, 
                                                  const float& min_det_conf,
                                                  std::vector<Detections>& out_detections) const {
    /** Store input bbox scores **/
    std::vector<float> scores(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        scores[i] = detections[i].detection_conf;
    }

    /** Estimate maximum number of algorithm iterations **/
    size_t max_queue_size = top_k > INVALID_TOP_K_IDX
                               ? std::min(static_cast<size_t>(top_k), scores.size())
                               : scores.size();

    /** Select top-k score indices **/
    std::vector<size_t> score_idx(scores.size());
    std::iota(score_idx.begin(), score_idx.end(), 0);
    std::partial_sort(score_idx.begin(), score_idx.begin() + max_queue_size, score_idx.end(),
        [&scores](size_t i1, size_t i2) {return scores[i1] > scores[i2];});

    /** Extract top-k score values **/
    std::vector<size_t> valid_score_idx(score_idx.begin(), score_idx.begin() + max_queue_size);
    std::vector<float> valid_scores(max_queue_size);
    for (size_t i = 0; i < valid_score_idx.size(); ++i) {
        valid_scores[i] = scores[valid_score_idx[i]];
    }

    /** Carry out Soft Non-Maximum Suppression algorithm **/
    std::vector<int> out_indices;
    out_indices.clear();
    for (size_t step = 0; step < valid_scores.size(); ++step) {
        auto best_score_itr = std::max_element(valid_scores.begin(), valid_scores.end());
        if (*best_score_itr < min_det_conf) {
            break;
        }

        /** Add current bbox to output list **/
        const size_t local_anchor_idx = std::distance(valid_scores.begin(), best_score_itr);
        const int anchor_idx = valid_score_idx[local_anchor_idx];
        out_indices.emplace_back(anchor_idx);
        *best_score_itr = 0.f;

        /** Update valid_scores of the rest bboxes **/
        for (size_t local_reference_idx = 0; local_reference_idx < valid_scores.size(); ++local_reference_idx) {
            /** Skip updating step for the low-confidence bbox **/
            if (valid_scores[local_reference_idx] < min_det_conf) {
                continue;
            }

            /** Calculate the Intersection over Union metric between two bboxes**/
            const size_t reference_idx = valid_score_idx[local_reference_idx];
            const auto& rect1 = detections[anchor_idx].rect;
            const auto& rect2 = detections[reference_idx].rect;
            const auto intersection = rect1 & rect2;
            float overlap = 0.f;
            if (intersection.width > 0 && intersection.height > 0) {
                const int intersection_area = intersection.area();
                overlap = static_cast<float>(intersection_area) / static_cast<float>(rect1.area() + rect2.area() - intersection_area);
            }

            /** Scale bbox score using the exponential rule **/
            valid_scores[local_reference_idx] *= std::exp(-overlap * overlap / sigma);
        }
    }
    for (size_t i = 0; i < out_indices.size(); ++i) {
        out_detections.emplace_back(detections[out_indices[i]]);
    }
}
inline ActionDetector::Bbox ActionDetector::ParseBBoxRecord(const float* data) const {
    Bbox box;
    box.xmin = data[0];
    box.ymin = data[1];
    box.xmax = data[2];
    box.ymax = data[3];
    return box;
}
