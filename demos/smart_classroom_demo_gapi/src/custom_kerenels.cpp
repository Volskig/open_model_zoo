// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom_kernels.hpp"
#include "kernel_packages.hpp"

/** OCV KERNELs */
GAPI_OCV_KERNEL(OCVFaceDetectorPostProc, custom::FaceDetectorPostProc) {
    static void run( const cv::Mat &in_ssd_result
                   , const cv::Mat &in_frame
                   , const detection::DetectorConfig & face_config
                   , std::vector<detection::DetectedObject> &out_faces) {
        std::unique_ptr<detection::FaceDetection> face_detector(new detection::FaceDetection(face_config));
        out_faces = face_detector->fetchResults(in_ssd_result, in_frame);
    }
};

GAPI_OCV_KERNEL(OCVGetRectFromImage, custom::GetRectFromImage) {
    static void run( const cv::Mat &in_image
                   , std::vector<cv::Rect> &out_rects) {
        out_rects.emplace_back(cv::Rect(0, 0, in_image.cols, in_image.rows));
    }
};

GAPI_OCV_KERNEL(OCVGetRectsFromDetections, custom::GetRectsFromDetections) {
    static void run( const detection::DetectedObjects &detections
                   , std::vector<cv::Rect> &out_rects) {
        for (const auto& it : detections) {
            out_rects.emplace_back(it.rect);
        }
    }
};

GAPI_OCV_KERNEL( OCVAlignFacesForReidentification
               , custom::AlignFacesForReidentification) {
    static void run( const std::vector<cv::Mat> &landmarks
                   , const std::vector<cv::Rect> &face_rois
                   , const cv::Mat &in_image
                   , cv::Mat &out_image) {
        in_image.copyTo(out_image);
        std::vector<cv::Mat> out_images;
        for (const auto& rect : face_rois) {
            out_images.emplace_back(out_image(rect));
        }
        std::vector<cv::Mat> landmarks_vec;
        for (const auto& el : landmarks) {
            landmarks_vec.emplace_back(el.reshape(1, { 5, 2 }));
        }
        AlignFaces(&out_images, &landmarks_vec);
        for (size_t i = 0; i < face_rois.size(); ++i) {
            out_images.at(i).copyTo(out_image(face_rois.at(i)));
        }
    }
};

GAPI_OCV_KERNEL( OCVPersonDetActionRecPostProc
               , custom::PersonDetActionRecPostProc) {
    static void run( const cv::Mat &in_ssd_local
                   , const cv::Mat &in_ssd_conf
                   , const cv::Mat &in_ssd_priorbox
                   , const cv::Mat &in_ssd_anchor1
                   , const cv::Mat &in_ssd_anchor2
                   , const cv::Mat &in_ssd_anchor3
                   , const cv::Mat &in_ssd_anchor4
                   , const cv::Mat &in_frame
                   , const ActionDetectorConfig & action_config
                   , DetectedActions &out_detections) {
        std::unique_ptr<ActionDetection> action_detector(new ActionDetection(action_config));
        out_detections = action_detector->fetchResults( { in_ssd_local
                                                        , in_ssd_conf
                                                        , in_ssd_priorbox
                                                        , in_ssd_anchor1
                                                        , in_ssd_anchor2
                                                        , in_ssd_anchor3
                                                        , in_ssd_anchor4 }
                                                      , in_frame);
    }
};

/** NOTE: Three OCV KERNELs for empty {} GArray **/
GAPI_OCV_KERNEL(OCVGetEmptyFaces, custom::GetEmptyFaces) {
    static void run(const cv::Mat &in,
        std::vector<detection::DetectedObject> &empty_faces) {
        empty_faces = {};
    }
};

GAPI_OCV_KERNEL(OCVGetEmptyActions, custom::GetEmptyActions) {
    static void run(const cv::Mat &in,
        std::vector<DetectedAction> &empty_actions) {
        empty_actions = {};
    }
};

GAPI_OCV_KERNEL(OCVGetEmptyMatGArray, custom::GetEmptyMatGArray) {
    static void run(const cv::Mat &in,
        std::vector<cv::Mat> &empty_garray) {
        empty_garray = {};
    }
};

cv::gapi::GKernelPackage kp::gallery_kernels() {
    return cv::gapi::kernels< OCVFaceDetectorPostProc
                            , OCVAlignFacesForReidentification
                            , OCVGetRectFromImage
                            , OCVGetRectsFromDetections>();
}

cv::gapi::GKernelPackage kp::video_process_kernels() {
    return cv::gapi::kernels < OCVFaceDetectorPostProc
                             , OCVGetRectsFromDetections
                             , OCVPersonDetActionRecPostProc
                             // , custom::OCVGetRecognizeResult
                             , OCVAlignFacesForReidentification
                             , OCVGetEmptyFaces
                             , OCVGetEmptyActions
                             , OCVGetEmptyMatGArray>();
}

cv::gapi::GKernelPackage kp::top_k_kernels() {
    return cv::gapi::kernels < OCVPersonDetActionRecPostProc
                             // , OCVGetActionTopHandsDetectionResult
                             , OCVGetEmptyActions>();
}
