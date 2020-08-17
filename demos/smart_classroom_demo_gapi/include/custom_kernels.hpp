// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "action_detector.hpp"
#include "face_reid.hpp"
#include "recognizer.hpp"
#include "tracker.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace state {
    /** NOTE: Questionable solution
      * kernels use num_frames,
      * num_frames total state need for FPS counter
      */
    extern size_t work_num_frames, total_num_frames;
}

template<> struct cv::detail::CompileArgTag<TrackerParams> {
    static const char* tag() {
        return "custom.get_action_detection_result_state_params";
    }
};

template<> struct cv::detail::CompileArgTag<TrackerParamsPack> {
    static const char* tag() {
        return "custom.get_recognition_result_state_params";
    }
};

namespace custom {
    G_API_OP( FaceDetectorPostProc
            , <cv::GArray<detection::DetectedObject>( cv::GMat
                                                    , cv::GMat
                                                    , detection::FaceDetectionKernelInput)>
            , "custom.fd_postproc") {
        static cv::GArrayDesc outMeta( const cv::GMatDesc &
                                     , const cv::GMatDesc &
                                     , const detection::FaceDetectionKernelInput &) {
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
                                         , ActionDetectionKernelInput)>
            , "custom.person_detection_action_recognition_postproc") {
        static cv::GArrayDesc outMeta( const cv::GMatDesc &, const cv::GMatDesc &
                                     , const cv::GMatDesc &, const cv::GMatDesc &
                                     , const cv::GMatDesc &, const cv::GMatDesc &
                                     , const cv::GMatDesc &, const cv::GMatDesc &
                                     , const ActionDetectionKernelInput &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( AlignFacesForReidentification
            , <cv::GMat( cv::GMat
                       , cv::GArray<cv::GMat>
                       , cv::GArray<cv::Rect>)>
            , "custom.align_faces_for_reidentification") {
        static cv::GMatDesc outMeta( const cv::GMatDesc & in
                                   , const cv::GArrayDesc &
                                   , const cv::GArrayDesc &) {
            return in;
        }
    };

    template<typename TO, typename S, typename TR> using rout = std::tuple<TO, TO, S, S, TR>;
    template<typename T> using tuple_five = std::tuple<T, T, T, T, T>;
    using RECInfo = rout< cv::GArray<TrackedObject>, cv::GArray<std::string>, cv::GArray<Track>>;

    G_API_OP( GetRecognitionResult
            , <RECInfo( cv::GMat
                      , cv::GArray<detection::DetectedObject>
                      , cv::GArray<DetectedAction>
                      , cv::GArray<cv::GMat>
                      , FaceRecognizerKernelInput)>
            , "custom.get_recognition_result") {
        static tuple_five <cv::GArrayDesc> outMeta( const cv::GMatDesc &
                                                  , const cv::GArrayDesc &
                                                  , const cv::GArrayDesc &
                                                  , const cv::GArrayDesc &
                                                  , const FaceRecognizerKernelInput &) {
            return std::make_tuple( cv::empty_array_desc()
                                  , cv::empty_array_desc()
                                  , cv::empty_array_desc()
                                  , cv::empty_array_desc()
                                  , cv::empty_array_desc());
        }
    };

    G_API_OP( GetActionTopHandsDetectionResult
            , <cv::GArray<TrackedObject>( cv::GMat
                                        , cv::GArray<DetectedAction>)>
            , "custom.get_action_detection_result_for_top_k_first_hands") {
        static cv::GArrayDesc outMeta( const cv::GMatDesc &
                                     , const cv::GArrayDesc &) {
            return cv::empty_array_desc();
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
