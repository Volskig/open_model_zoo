// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "custom_kernels.hpp"
#include "smart_classroom_demo.hpp"
#include "actions.hpp"
#include "kernel_packages.hpp"

#include <opencv2/gapi/infer/ie.hpp>

namespace nets {
    G_API_NET(FaceDetector,        <cv::GMat(cv::GMat)>, "face-detector");
    G_API_NET(LandmarksDetector,   <cv::GMat(cv::GMat)>, "landmarks-detector");
    G_API_NET(FaceReidentificator, <cv::GMat(cv::GMat)>, "face-reidentificator");
    using PAInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat>;
    G_API_NET(PersonDetActionRec,  <PAInfo(cv::GMat)>,   "person-detection-action-recognition");
} // namespace nets

namespace config {
const std::array<std::string, 7> action_detector_5 = {
    "mbox_loc1/out/conv/flat",
    "mbox_main_conf/out/conv/flat/softmax/flat",
    "mbox/priorbox",
    "out/anchor1",
    "out/anchor2",
    "out/anchor3",
    "out/anchor4"};

const std::array<std::string, 7> action_detector_6 = {
    "ActionNet/out_detection_loc",
    "ActionNet/out_detection_conf",
    "ActionNet/action_heads/out_head_1_anchor_1",
    "ActionNet/action_heads/out_head_2_anchor_1",
    "ActionNet/action_heads/out_head_2_anchor_2",
    "ActionNet/action_heads/out_head_2_anchor_3",
    "ActionNet/action_heads/out_head_2_anchor_4"};;

inline char separator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

bool fileExists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

std::string folderName(const std::string& path) {
    size_t found_pos;
    found_pos = path.find_last_of(separator());
    if (found_pos != std::string::npos)
        return path.substr(0, found_pos);
    return std::string(".") + separator();
}

std::vector<std::string> parseActionLabels(const std::string& in_str) {
    std::vector<std::string> labels;
    std::string label;
    std::istringstream stream_to_split(in_str);
    while (std::getline(stream_to_split, label, ',')) {
        labels.push_back(label);
    }
    return labels;
}

FaceRecognizerConfig getRecConfig() {
    FaceRecognizerConfig rec_config;
    rec_config.reid_threshold = FLAGS_t_reid;
    rec_config.greedy_reid_matching = FLAGS_greedy_reid_matching;
    return rec_config;
}

bool isNetForSixActions(const std::string& model_path) {
    CV_Assert(!model_path.empty());
    return model_path.at(model_path.size() - 5) == '6';
}

void createActDetPtr(const bool net_with_six_actions,
                     const cv::Size frame_size,
                     const size_t actions_map_size,
                     ActionDetectionKernelInput& ad_kernel_input) {
    // Load action detector
    ActionDetectorConfig action_config;
    action_config.net_with_six_actions = net_with_six_actions;
    action_config.detection_confidence_threshold = static_cast<float>(FLAGS_t_ad);
    action_config.action_confidence_threshold = static_cast<float>(FLAGS_t_ar);
    action_config.num_action_classes = actions_map_size;
    action_config.input_height = frame_size.height;
    action_config.input_width = frame_size.width;
    ad_kernel_input.ptr.reset(new ActionDetection(action_config));
}

detection::DetectorConfig getDetConfig() {
    // Load face detector
    detection::DetectorConfig face_config;
    face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
    face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
    face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);
    return face_config;
}

void createFaceDetPtr(detection::FaceDetectionKernelInput& fd_kernel_input) {
    const auto face_det_config = getDetConfig();
    fd_kernel_input.ptr.reset(new detection::FaceDetection(face_det_config));
}

void createFaceRegPtr(detection::FaceDetectionKernelInput& fd_kernel_input) {
    auto face_registration_det_config = getDetConfig();
    face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
    fd_kernel_input.ptr.reset(new detection::FaceDetection(face_registration_det_config));
}

void createFaceRecPtr(const FaceRecognizerConfig& rec_config,
                            FaceRecognizerKernelInput& frec_kernel_input) {
    frec_kernel_input.ptr.reset(new FaceRecognizer(rec_config));
}

ConstantParams getConstants(const std::string& video_path, const cv::Size frame_size,
                            const int fps, const size_t num_frames) {
    ConstantParams const_params;
    const_params.teacher_id = FLAGS_teacher_id;
    const_params.actions_type = FLAGS_teacher_id.empty()
                                        ? FLAGS_a_top > 0 ? TOP_K : STUDENT
                                        : TEACHER;
        const_params.actions_map = const_params.actions_type == STUDENT
                                       ? parseActionLabels(FLAGS_student_ac)
                                       : const_params.actions_type == TOP_K
                                           ? parseActionLabels(FLAGS_top_ac)
                                           : parseActionLabels(FLAGS_teacher_ac);
    const_params.top_action_id =
            static_cast<int>(const_params.actions_type == TOP_K
                                ? std::distance(const_params.actions_map.begin(),
                                                find(const_params.actions_map.begin(), const_params.actions_map.end(), FLAGS_top_id))
                                : -1);

    if (const_params.actions_type == TOP_K &&
        (const_params.top_action_id < 0 || const_params.top_action_id >= static_cast<int>(const_params.actions_map.size()))) {
        slog::err << "Cannot find target action: " << FLAGS_top_id << slog::endl;
    }
    const auto num_top_persons = const_params.actions_type == TOP_K ? FLAGS_a_top : -1;
    const_params.draw_ptr.reset(new DrawingHelper(FLAGS_no_show, num_top_persons));
    const_params.num_frames = static_cast<size_t>(FLAGS_limit < num_frames
        ? FLAGS_limit
        : num_frames);
    const_params.video_path = video_path;
    slog::info << "Reading video '" << video_path << "'" << slog::endl;
    const_params.smooth_window_size = fps * FLAGS_d_ad;
    const_params.smooth_min_length = fps * FLAGS_min_ad;
    const_params.top_flag = FLAGS_a_top;
    const_params.draw_ptr->GetNewFrameSize(frame_size);
    return const_params;
}

void printInfo() {
    if (!FLAGS_teacher_id.empty() && !FLAGS_top_id.empty()) {
        slog::err << "Cannot run simultaneously teacher action and top-k students recognition."
                  << slog::endl;
    }
    InferenceEngine::Core ie;
    std::vector<std::string> devices = {FLAGS_d_act, FLAGS_d_fd, FLAGS_d_lm, FLAGS_d_reid};
    std::set<std::string> loadedDevices;
    slog::info << "Device info: " << slog::endl;
    for (const auto &device : devices) {
        if (loadedDevices.find(device) != loadedDevices.end())
            continue;
        std::cout << printable(ie.GetVersions(device)) << std::endl;
        loadedDevices.insert(device);
    }
}

std::string GetBinPath(const std::string& pathXML) {
    CV_Assert(pathXML.substr(pathXML.size() - 4, pathXML.size()) == ".xml");
    std::string pathBIN(pathXML);
    return pathBIN.replace(pathBIN.size() - 3, 3, "bin");
}

void configNets(const std::string& fd_model_path,
                const std::string& lm_model_path,
                const std::string& fr_model_path,
                const std::string& ad_model_path,
                      cv::gapi::ie::Params<nets::FaceDetector>&        det_net,
                      cv::gapi::ie::Params<nets::LandmarksDetector>&   landm_net,
                      cv::gapi::ie::Params<nets::FaceReidentificator>& reident_net,
                      cv::gapi::ie::Params<nets::PersonDetActionRec>&  action_net) {
    if (!ad_model_path.empty()) {
       /** Create action detector net's parameters **/
       std::array<std::string, 7> outputBlobList;
          outputBlobList = isNetForSixActions(ad_model_path)
              ? outputBlobList = config::action_detector_6
              : outputBlobList =  config::action_detector_5;
       action_net = cv::gapi::ie::Params<nets::PersonDetActionRec>{
           ad_model_path,
           GetBinPath(ad_model_path),
           FLAGS_d_act,
       }.cfgOutputLayers(outputBlobList);
    }
    if (!fd_model_path.empty()) {
        /** Create face detector net's parameters **/
           det_net = cv::gapi::ie::Params<nets::FaceDetector>{
               fd_model_path,
               GetBinPath(fd_model_path),
               FLAGS_d_fd,
           }.cfgInputReshape("data",
                            {1u, 3u, static_cast<size_t>(FLAGS_inh_fd), static_cast<size_t>(FLAGS_inw_fd)});
    }
    if (!fd_model_path.empty() && !fr_model_path.empty() && !lm_model_path.empty()) {
        /** Create landmarks detector net's parameters **/
        landm_net = cv::gapi::ie::Params<nets::LandmarksDetector>{
            lm_model_path,
            GetBinPath(lm_model_path),
            FLAGS_d_lm,
        };
        /** Create reidentification net's parameters **/
        reident_net = cv::gapi::ie::Params<nets::FaceReidentificator>{
            fr_model_path,
            GetBinPath(fr_model_path),
            FLAGS_d_reid,
        };
    }
}
} // namespace config

namespace preparation {
void processingFaceGallery(const cv::gapi::ie::Params<nets::FaceDetector>&        face_net,
                           const cv::gapi::ie::Params<nets::LandmarksDetector>&   landm_net,
                           const cv::gapi::ie::Params<nets::FaceReidentificator>& reident_net,
                                 FaceRecognizerKernelInput& frec_kernel_input,
                                 std::vector<std::string>&  face_id_to_label_map) {
    // Face gallery processing
    std::vector<int> idx_to_id;
    std::vector<GalleryObject> identities;
    const auto ids_list = FLAGS_fg;
    detection::FaceDetectionKernelInput reid_kernel_input;
    config::createFaceRegPtr(reid_kernel_input);
    if (!ids_list.empty()) {
        /** Gallery graph of demo **/
        /** Input is one face from gallery **/
        cv::GMat in;
        cv::GArray<cv::Rect> rect;

        /** Crop face from image **/
        if (FLAGS_crop_gallery) {
            /** Detect face **/
            cv::GMat detections =
                cv::gapi::infer<nets::FaceDetector>(in);

            cv::GArray<detection::DetectedObject> faces =
                custom::FaceDetectorPostProc::on(in,
                                                 detections,
                                                 reid_kernel_input);
            /** Get ROI for face **/
            rect = custom::GetRectsFromDetections::on(faces);
        } else {
            /** Else ROI is equal to image size **/
            rect = custom::GetRectFromImage::on(in);
        }
        /** Get landmarks by ROI **/
        cv::GArray<cv::GMat> landmarks =
            cv::gapi::infer<nets::LandmarksDetector>(rect, in);

        /** Align face by landmarks **/
        cv::GArray<cv::GMat> align_faces =
            custom::AlignFacesForReidentification::on(in, landmarks, rect);

        /** Get face identities metrics **/
        cv::GArray<cv::GMat> embeddings = cv::gapi::infer2<nets::FaceReidentificator>(in, align_faces);

        /** Pipeline's input and outputs**/
        cv::GComputation gallery_pp(cv::GIn(in), cv::GOut(rect, embeddings));

        auto gallery_networks = cv::gapi::networks(face_net, landm_net, reident_net);

        cv::FileStorage fs(ids_list, cv::FileStorage::Mode::READ);
        cv::FileNode fn = fs.root();
        int id = 0;
        for (const auto& item : fn) {
            std::string label = item.name();
            std::vector<cv::Mat> out_embeddings;
            // Please, note that the case when there are more than one image in gallery
            // for a person might not work properly with the current implementation
            // of the demo.
            // Remove this assert by your own risk.
            CV_Assert(item.size() == 1);

            for (const auto& item_e : item) {
                cv::Mat image;
                std::vector<cv::Mat> emb;
                if (config::fileExists(item_e.string())) {
                    image = cv::imread(item_e.string());
                } else {
                    image = cv::imread(config::folderName(ids_list) + config::separator() + item_e.string());
                }
                CV_Assert(!image.empty());
                std::vector<cv::Rect> out_rect;
                gallery_pp.apply(cv::gin(image), cv::gout(out_rect, emb),
                                 cv::compile_args(custom::kernels(), gallery_networks));
                CV_Assert(emb.size() == 1);
                CV_Assert(out_rect.size() == 1);
                // NOTE: RegistrationStatus analog check
                if (!out_rect.empty() &&
                    !(out_rect.size() > 1) &&
                    (out_rect[0].width > FLAGS_min_size_fr) &&
                    (out_rect[0].height > FLAGS_min_size_fr)) {
                    out_embeddings.emplace_back(emb.front().reshape(1, { 256, 1 }));
                    idx_to_id.emplace_back(id);
                    identities.emplace_back(out_embeddings, label, id++);
                }
            }
        }
        slog::info << "Face reid gallery size: " << identities.size() << slog::endl;
    } else {
        slog::warn << "Face reid gallery is empty!" << slog::endl;
    }
    auto rec_config = config::getRecConfig();
    rec_config.identities = identities;
    rec_config.idx_to_id = idx_to_id;
    config::createFaceRecPtr(rec_config, frec_kernel_input);
    face_id_to_label_map = frec_kernel_input.ptr->GetIDToLabelMap();
}
} // namespace preparation