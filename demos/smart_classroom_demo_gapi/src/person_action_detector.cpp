// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "person_action_detector.hpp"
#include <algorithm>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#define SSD_LOCATION_RECORD_SIZE 4
#define SSD_PRIORBOX_RECORD_SIZE 4
#define NUM_DETECTION_CLASSES 2
#define POSITIVE_DETECTION_IDX 1
#define NUM_CANDIDATES 4300
#define INVALID_TOP_K_IDX -1

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

struct Bbox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
};

inline cv::Rect ConvertToRect(const Bbox& prior_bbox, const Bbox& variances,
        const Bbox& encoded_bbox, const cv::Size& frame_size) {
    /** Convert prior bbox to CV_Rect **/
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

inline void SoftNonMaxSuppression(const std::vector<Detections>& detections,
                           const float& sigma, const int& top_k, 
                           const float& min_det_conf,
                           std::vector<Detections>& out_detections) {
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
inline Bbox ParseBBoxRecord(const float* data) {
    Bbox box;
    box.xmin = data[0];
    box.ymin = data[1];
    box.xmax = data[2];
    box.ymax = data[3];
    return box;
}
namespace custom {

}
