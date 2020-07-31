// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "action_detector.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "recognizer.hpp"
#include "tracker.hpp"
#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace {
    detection::DetectorConfig face_config;
    detection::DetectorConfig face_registration_det_config;
    ActionDetectorConfig action_config;
    FaceRecognizerConfig rec_config;
} // namespace 

namespace custom {
    /** OPs */
    G_API_OP( FaceDetectorPostProc
            , <cv::GArray<detection::DetectedObject>( cv::GMat
                                                    , cv::GMat
                                                    , detection::DetectorConfig)>
            , "custom.fd_postproc") {
        static cv::GArrayDesc outMeta( const cv::GMatDesc &
                                     , const cv::GMatDesc &
                                     , const detection::DetectorConfig &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( GetRectFromImage
            , <cv::GArray<cv::Rect>(cv::GMat)>
            , "custom.get_rect_from_image") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( GetRectsFromDetections
            , <cv::GArray<cv::Rect>(cv::GArray<detection::DetectedObject>)>
            , "custom.get_rects_from_detection") {
        static cv::GArrayDesc outMeta(const cv::GArrayDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( PersonDetActionRecPostProc
            , <cv::GArray<DetectedAction>( cv::GMat, cv::GMat
                                         , cv::GMat, cv::GMat
                                         , cv::GMat, cv::GMat
                                         , cv::GMat, cv::GMat
                                         , ActionDetectorConfig)>,
        "custom.person_detection_action_recognition_postproc") {
        static cv::GArrayDesc outMeta( const cv::GMatDesc &, const cv::GMatDesc &
                                     , const cv::GMatDesc &, const cv::GMatDesc &
                                     , const cv::GMatDesc &, const cv::GMatDesc &
                                     , const cv::GMatDesc &, const cv::GMatDesc &
                                     , const ActionDetectorConfig &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( AlignFacesForReidentification
            , <cv::GMat( cv::GArray<cv::GMat>
                       , cv::GArray<cv::Rect>
                       , cv::GMat)>
            , "custom.align_faces_for_reidentification") {
        static cv::GMatDesc outMeta( const cv::GArrayDesc &
                                   , const cv::GArrayDesc &
                                   , const cv::GMatDesc & in) {
            return in;
        }
    };

    /** NOTE: Three crutch-Ops for empty {} GArray **/
    G_API_OP( GetEmptyFaces
            , <cv::GArray<detection::DetectedObject>(cv::GMat)>
            , "custom.get_empty_faces_garray") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( GetEmptyActions
            , <cv::GArray<DetectedAction>(cv::GMat)>
            , "custom.get_empty_actions_garray") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( GetEmptyMatGArray
            , <cv::GArray<cv::GMat>(cv::GMat)>
            , "custom.get_empty_garray_of_mat") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };
}
