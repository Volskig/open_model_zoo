// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "kernel_packages.hpp"
#include "custom_kernels.hpp"

/** OCV KERNELs */
GAPI_OCV_KERNEL(OCVFaceDetectorPostProc, custom::FaceDetectorPostProc) {
    static void run( const cv::Mat &in_frame
                   , const cv::Mat &in_ssd_result
                   , const detection::FaceDetectionKernelInput & face_inp
                   , std::vector<detection::DetectedObject> &out_faces) {
        out_faces = face_inp.ptr->fetchResults(in_ssd_result, in_frame);
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
    static void run( const cv::Mat &in_image
                   , const std::vector<cv::Mat> &landmarks
                   , const std::vector<cv::Rect> &face_rois
                   , cv::Mat &out_image) {
        in_image.copyTo(out_image);
        std::vector<cv::Mat> out_images;
        for (const auto& rect : face_rois) {
            out_images.emplace_back(out_image(rect));
        }
        /** NOTE: AlignFaces works with shape 5:2 after G-API we get 1:10:1:1
          * const_cast instead creation new non-const vector, reshape in AlignFaces()
          */
        AlignFaces(&out_images, &const_cast<std::vector<cv::Mat>&>(landmarks));
        for (size_t i = 0; i < face_rois.size(); ++i) {
            out_images.at(i).copyTo(out_image(face_rois.at(i)));
        }
    }
};

GAPI_OCV_KERNEL( OCVPersonDetActionRecPostProc
               , custom::PersonDetActionRecPostProc) {
    static void run( const cv::Mat &in_frame
                   , const cv::Mat &in_ssd_local
                   , const cv::Mat &in_ssd_conf
                   , const cv::Mat &in_ssd_priorbox
                   , const cv::Mat &in_ssd_anchor1
                   , const cv::Mat &in_ssd_anchor2
                   , const cv::Mat &in_ssd_anchor3
                   , const cv::Mat &in_ssd_anchor4
                   , const ActionDetectionKernelInput & action_in
                   , DetectedActions &out_detections) {
        out_detections = action_in.ptr->fetchResults( { in_ssd_local
                                                        , in_ssd_conf
                                                        , in_ssd_priorbox
                                                        , in_ssd_anchor1
                                                        , in_ssd_anchor2
                                                        , in_ssd_anchor3
                                                        , in_ssd_anchor4 }
                                                        , in_frame);
    }
};

GAPI_OCV_KERNEL_ST( OCVGetActionTopHandsDetectionResult
                  , custom::GetActionTopHandsDetectionResult
                  , Tracker) {
    static void setup( const cv::GMatDesc & 
                     , const cv::GArrayDesc &
                     , std::shared_ptr<Tracker> &tracker_action
                     , const cv::GCompileArgs &compileArgs) {
        auto trParam = cv::gapi::getCompileArg<TrackerParams>(compileArgs)
            .value_or(TrackerParams{});
        tracker_action = std::make_shared<Tracker>(trParam);
    }
    static void run( const cv::Mat &frame
                   , const DetectedActions & actions
                   , TrackedObjects &tracked_actions
                   , Tracker &tracker_action) {
        TrackedObjects tracked_action_objects;
        for (const auto& action : actions) {
            tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
        }
        tracker_action.Process(frame, tracked_action_objects, state::total_num_frames);
        tracked_actions = tracker_action.TrackedDetectionsWithLabels();
    }
};

GAPI_OCV_KERNEL_ST(OCVGetRecognitionResult, custom::GetRecognitionResult, TrackersPack) {
    static void setup( const cv::GMatDesc &
                     , const cv::GArrayDesc &
                     , const cv::GArrayDesc &
                     , const cv::GArrayDesc &
                     , const FaceRecognizerKernelInput &
                     , std::shared_ptr<TrackersPack> &trackers
                     , const cv::GCompileArgs &compileArgs) {
        auto trParamsPack = cv::gapi::getCompileArg<TrackerParamsPack>(compileArgs)
            .value_or(TrackerParamsPack{});
        trackers = std::make_shared<TrackersPack>(trParamsPack.tracker_reid_params,
            trParamsPack.tracker_action_params);
    }
    static void run( const cv::Mat &frame
                   , const detection::DetectedObjects &faces
                   , const DetectedActions & actions
                   , const std::vector<cv::Mat>& embeddings
                   , const FaceRecognizerKernelInput &rec_in
                   , TrackedObjects &tracked_faces
                   , TrackedObjects &tracked_actions
                   , std::vector<std::string> &face_labels
                   , std::vector<std::string> &face_id_to_label_map
                   , std::vector<Track> &face_tracks
                   , TrackersPack &trackers) {
        TrackedObjects tracked_face_objects;
        TrackedObjects tracked_action_objects;
        /** NOTE: Recognize works with shape 256:1 after G-API we get 1:256:1:1
          * const_cast instead creation new non-const vector, reshape in Recognize()
          */
        std::vector<int> ids = rec_in.ptr->Recognize(const_cast<std::vector<cv::Mat>&>(embeddings), faces);
        for (size_t i = 0; i < faces.size(); ++i) {
            tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
        }
        trackers.tracker_reid.Process(frame, tracked_face_objects, state::work_num_frames);
        tracked_faces = trackers.tracker_reid.TrackedDetectionsWithLabels();

        for (const auto& face : tracked_faces) {
            face_labels.push_back(rec_in.ptr->GetLabelByID(face.label));
        }

        for (const auto& action : actions) {
            tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
        }
        trackers.tracker_action.Process(frame, tracked_action_objects, state::work_num_frames);
        tracked_actions = trackers.tracker_action.TrackedDetectionsWithLabels();

        if (!rec_in.actions_type) {
            face_tracks = trackers.tracker_reid.vector_tracks();
            face_id_to_label_map = rec_in.ptr->GetIDToLabelMap();
        }
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
                             , OCVAlignFacesForReidentification
                             , OCVGetRecognitionResult
                             , OCVGetEmptyFaces
                             , OCVGetEmptyActions
                             , OCVGetEmptyMatGArray>();
}

cv::gapi::GKernelPackage kp::top_k_kernels() {
    return cv::gapi::kernels < OCVPersonDetActionRecPostProc
                             , OCVGetActionTopHandsDetectionResult
                             , OCVGetEmptyActions>();
}
