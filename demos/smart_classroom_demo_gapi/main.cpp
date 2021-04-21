// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <monitors/presenter.h>
#include <ie_iextension.h>

#include <opencv2/gapi/render.hpp>
#include <opencv2/gapi/core.hpp>

#include "initialize.hpp"
#include "stream_source.hpp"

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;
    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }
    if (FLAGS_m_act.empty() && FLAGS_m_fd.empty()) {
        throw std::logic_error("At least one parameter -m_act or -m_fd must be set");
    }
    return true;
}

int main(int argc, char* argv[]) { 
    try {
        /** This demo covers 4 certain topologies and cannot be generalized **/
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** Prepare parameters **/
        const std::string video_path = FLAGS_i;
        const auto ad_model_path     = FLAGS_m_act;
        const auto fd_model_path     = FLAGS_m_fd;
        const auto fr_model_path     = FLAGS_m_reid;
        const auto lm_model_path     = FLAGS_m_lm;

        /** Print info about demo's properties **/
        config::printInfo();

        cv::VideoCapture cap(video_path != "cam" ? video_path : 0);

        /** Get information about frame from cv::VideoCapture **/
        const auto frame_size = cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                                         static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));

        /** Fill shared constants and tracker parameters **/
        TrackerParams tracker_reid_params, tracker_action_params;
        ConstantParams const_params;
        std::tie(const_params, tracker_reid_params, tracker_action_params) =
            config::getGraphArgs(video_path,
                                 frame_size,
                                 static_cast<int>(cap.get(cv::CAP_PROP_FPS)),
                                 static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)));

        /** Create default net's parameters **/
        cv::gapi::ie::Params<nets::FaceDetector> det_net({}, {}, {});
        cv::gapi::ie::Params<nets::LandmarksDetector> landm_net({}, {}, {});
        cv::gapi::ie::Params<nets::FaceReidentificator> reident_net({}, {}, {});
        cv::gapi::ie::Params<nets::PersonDetActionRec> action_net({}, {}, {});

         /** Configure nets **/
        config::configNets(fd_model_path, lm_model_path, fr_model_path, ad_model_path,
                           det_net,       landm_net,     reident_net,   action_net);
        auto networks = cv::gapi::networks(det_net, landm_net, reident_net, action_net);

        /** Configure and create action detector **/
        ActionDetectionKernelInput ad_kernel_input;
        if (!ad_model_path.empty()) {
            config::createActDetPtr(config::isNetForSixActions(ad_model_path),
                                    frame_size,
                                    const_params.actions_map.size(),
                                    ad_kernel_input);
        }

        /** Configure and create face detector **/
        detection::FaceDetectionKernelInput fd_kernel_input;
        if (!fd_model_path.empty()) {
            config::createFaceDetPtr(fd_kernel_input);
        }

        /** Find identities metric for each face from gallery **/
        FaceRecognizerKernelInput frec_kernel_input;
        std::vector<std::string> face_id_to_label_map;
        preparation::processingFaceGallery(det_net, landm_net, reident_net, frec_kernel_input, face_id_to_label_map);
        if (fd_model_path.empty() && fr_model_path.empty() && lm_model_path.empty()) {
            slog::warn << "Face recognition models are disabled!" << slog::endl;
            if (const_params.actions_type == TEACHER) {
                slog::err << "Face recognition must be enabled to recognize teacher actions." << slog::endl;
                return 1;
            }
        }
        if (const_params.actions_type == TEACHER && !frec_kernel_input.ptr->LabelExists(const_params.teacher_id)) {
            slog::err << "Teacher id does not exist in the gallery!" << slog::endl;
            return 1;
        }

        /** Main graph of demo **/
        cv::GMat in;
        cv::GMat pp_frame = cv::gapi::copy(in);
        /** Initialize empty GArrays **/
        cv::GArray<cv::GMat> embeddings(std::vector<cv::Mat>{});
        cv::GArray<DetectedAction> persons_with_actions(std::vector<DetectedAction>{});
        cv::GArray<cv::Rect> rects(std::vector<cv::Rect>{});

        if (const_params.actions_type != TOP_K) {
            if (!fd_model_path.empty()) {
                /** Face detection **/
                cv::GMat detections = cv::gapi::infer<nets::FaceDetector>(in);
                cv::GOpaque<cv::Size> sz = cv::gapi::streaming::size(in);
                cv::GArray<cv::Rect> face_rects;
                cv::GArray<int> labels;
                std::tie(face_rects, labels) = cv::gapi::parseSSD(detections, sz, float(FLAGS_t_fd), -1);
                rects = custom::FaceDetectorPostProc::on(in,
                                                         face_rects,
                                                         fd_kernel_input);
                if (!fr_model_path.empty() && !lm_model_path.empty()) {
                    /** Get landmarks **/
                    cv::GArray<cv::GMat> landmarks =
                        cv::gapi::infer<nets::LandmarksDetector>(rects, in);
                    /** Get aligned faces **/
                     cv::GArray<cv::GMat> align_faces =
                        custom::AlignFacesForReidentification::on(in, landmarks, rects);
                    /** Get face identities metrics for each person **/
                    embeddings = cv::gapi::infer2<nets::FaceReidentificator>(in, align_faces);
                }
            }
        }

        /** First graph output **/
        auto outs = GOut(pp_frame);
        if (!ad_model_path.empty()) {
            cv::GMat location, detect_confidences, priorboxes, action_con1, action_con2, action_con3, action_con4;
            /** Action detection-recognition **/
            std::tie(location, detect_confidences, priorboxes, action_con1, action_con2, action_con3, action_con4) =
                cv::gapi::infer<nets::PersonDetActionRec>(in);

            /** Get actions for each person on frame **/
            persons_with_actions =
                custom::PersonDetActionRecPostProc::on(in, location, detect_confidences,
                                                       priorboxes, action_con1,
                                                       action_con2, action_con3,
                                                       action_con4, ad_kernel_input);
        }
        cv::GOpaque<DrawingElements> draw_elements;
        cv::GArray<TrackedObject> tracked_actions;
        if (const_params.actions_type != TOP_K) {
            /** Main demo scenario **/
            cv::GOpaque<FaceTrack> face_track;
            cv::GOpaque<size_t> work_num_frames;
            /** Recognize actions and faces **/
            std::tie(tracked_actions, face_track, work_num_frames) =
                custom::GetRecognitionResult::on(in, rects, persons_with_actions, embeddings, frec_kernel_input, const_params);

            cv::GOpaque<std::string> stream_log, stat_log, det_log;
            cv::GArray<std::string> face_ids(face_id_to_label_map);
            /** Get roi and labels for drawing and set logs **/
            std::tie(draw_elements, stream_log, stat_log, det_log) =
                custom::RecognizeResultPostProc::on(in, tracked_actions, face_track, face_ids, work_num_frames, const_params);
            /** Main demo part of graph output **/
            outs += GOut(work_num_frames, stream_log, stat_log, det_log);
        } else {
            /** Top action case **/
            cv::GMat top_k;
            /** Recognize actions **/
            tracked_actions =
                custom::GetActionTopHandsDetectionResult::on(in, persons_with_actions);
             /** Get roi and labels for drawing **/
            std::tie(draw_elements, top_k) = custom::TopAction::on(in, tracked_actions, const_params);
            /** Top action case part of graph output **/
            outs += GOut(top_k);
        }
        /** Draw ROI and labels **/
        auto rendered = cv::gapi::wip::draw::render3ch(pp_frame,
                                                       custom::BoxesAndLabels::on(pp_frame, draw_elements, const_params));
        /** Last graph output is frame to draw **/
        outs += GOut(rendered);

        /** Pipeline's input and outputs**/
        cv::GComputation pp(cv::GIn(in), std::move(outs));

        cv::GStreamingCompiled cc = pp.compileStreaming(cv::compile_args(custom::kernels(),
                                                        networks,
                                                        TrackerParamsPack{ tracker_reid_params, tracker_action_params },
                                                        LoggerParams{ FLAGS_r }));

        /** The execution part **/
        cc.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::CustomCapSource>(cap)));

        /** Service constants **/
        float wait_time_ms = 0.f;
        float work_time_ms = 0.f;
        size_t wait_num_frames = 0;
        size_t work_num_frames = 0;
        size_t total_num_frames = 0;
        size_t work_time_ms_all = 0;
        const char SPACE_KEY = 32;
        const char ESC_KEY = 27;
        bool monitoring_enabled = const_params.actions_type == TOP_K ? false : true;
        cv::Size graphSize { static_cast<int>(frame_size.width / 4), 60 };

        /** Presenter for rendering system parameters **/
        Presenter presenter(FLAGS_u, frame_size.height - graphSize.height - 10, graphSize);

        /** Create VideoWriter **/
        cv::VideoWriter vid_writer;
        if (!FLAGS_out_v.empty() && !vid_writer.isOpened()) {
            vid_writer = cv::VideoWriter(FLAGS_out_v,
                                         cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                         cap.get(cv::CAP_PROP_FPS), frame_size);
        }

        /** Result containers associated with graph output **/
        cv::Mat frame, proc, top_k;
        std::string stream_log, stat_log, det_log;
        auto out_vector = cv::gout(frame);
        if (const_params.actions_type == TOP_K) {
            out_vector += cv::gout(top_k, proc);
        } else {
            out_vector += cv::gout(work_num_frames, stream_log, stat_log, det_log, proc);
        }

        /** TOP_K case starts without processing **/
        if (const_params.actions_type != TOP_K) cc.start();

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key";
        }
        std::cout << std::endl;

        /** Main cycle **/
        auto started_all = std::chrono::high_resolution_clock::now();
        while (true) {
            auto started = std::chrono::high_resolution_clock::now();
            char key = cv::waitKey(1);
            presenter.handleKey(key);
            if (key == ESC_KEY) {
                break;
            }
            if (const_params.actions_type == TOP_K) {
                if ((key == SPACE_KEY && !monitoring_enabled) ||
                    (key == SPACE_KEY && monitoring_enabled)) {
                    /** SPACE_KEY & monitoring_enabled trigger **/
                    monitoring_enabled = !monitoring_enabled;
                    const_params.draw_ptr->ClearTopWindow();
                }
            }
            if (monitoring_enabled) {
                if (!cc.running()) {
                    /** TOP_K part. SPACE_KEY is pushed, monitoring enabled
                     *  Compile and start graph **/
                    if (!cap.grab()) break;
                    cc.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::CustomCapSource>(cap)));
                    cc.start();
                }
                if (!cc.pull(std::move(out_vector))) {
                    /** Main part. Processing is always on **/
                    if (cv::waitKey(1) >= 0) break;
                    else continue;
                }
            } else {
                /** TOP_K part. monitoring isn't enabled **/
                if (cc.running()) cc.stop();
                /** Get clear frame **/
                if (!cap.read(frame)) break;
                const auto new_height = cvRound(frame.rows * const_params.draw_ptr->rect_scale_y_);
                const auto new_width = cvRound(frame.cols * const_params.draw_ptr->rect_scale_x_);
                cv::resize(frame, frame, cv::Size(new_width, new_height));
                auto elapsed = std::chrono::high_resolution_clock::now() - started;
                wait_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                const_params.draw_ptr->DrawFPS(frame, 1e3f / (wait_time_ms / static_cast<float>(++wait_num_frames) + 1e-6f),
                                               CV_RGB(0, 255, 0));
                presenter.drawGraphs(frame);
                const_params.draw_ptr->Show(frame);
                const_params.draw_ptr->ShowCrop();
            }
            if (const_params.actions_type == TOP_K && monitoring_enabled) {
                /** TOP_K part. monitoring is enabled and graph is started **/
                auto elapsed = std::chrono::high_resolution_clock::now() - started;
                work_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                const_params.draw_ptr->DrawFPS(proc, 1e3f / (work_time_ms / static_cast<float>(++work_num_frames) + 1e-6f),
                                               CV_RGB(255, 0, 0));
                const_params.draw_ptr->Show(proc);
                const_params.draw_ptr->ShowCrop(top_k);
                total_num_frames = work_num_frames + wait_num_frames;
            } else if (const_params.actions_type != TOP_K) {
                /** Main part. Processing is always on **/
                auto elapsed = std::chrono::high_resolution_clock::now() - started;
                work_time_ms += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
                if (!vid_writer.isOpened()) {
                    const_params.draw_ptr->DrawFPS(proc, 1e3f / (work_time_ms / static_cast<float>(work_num_frames) + 1e-6f),
                                                   CV_RGB(255, 0, 0));
                }
                presenter.drawGraphs(proc);
                const_params.draw_ptr->Show(proc);
                total_num_frames = work_num_frames;
            }
            if (vid_writer.isOpened()) {
                vid_writer << proc;
            };
            if (FLAGS_loop && (work_num_frames == const_params.num_frames)) {
                /** Loop **/
                cc.stop();
                cap.set(cv::CAP_PROP_POS_FRAMES, 0.);
                cc.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::CustomCapSource>(cap)));
                cc.start();
            }
            if (FLAGS_limit >= 0 && (work_num_frames > static_cast<size_t>(FLAGS_limit))) {
                /** Frame limit reached **/
                break;
            }
            /** Console log, if exists **/
            std::cout << stream_log;
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - started_all;
        work_time_ms_all += std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        if (vid_writer.isOpened()) {
            vid_writer.release();
        };
        const_params.draw_ptr->Finalize();
        /** Print logs to files **/
        std::ofstream act_stat_log_stream, act_det_log_stream;
        if (!FLAGS_al.empty()) {
                act_det_log_stream.open(FLAGS_al,  std::fstream::out);
                act_det_log_stream << "data" << "[" << std::endl;
                act_det_log_stream << det_log << "]";
        }
        act_stat_log_stream.open(FLAGS_ad, std::fstream::out);
        act_stat_log_stream << stat_log << std::endl;
        slog::info << slog::endl;
        /** Results **/
        if ( work_num_frames > 0) {
            const float mean_time_ms = work_time_ms_all / static_cast<float>(work_num_frames);
            slog::info << "Mean FPS: " << 1e3f / mean_time_ms << slog::endl;
        }
        slog::info << "Frames: " << total_num_frames << slog::endl;
        std::cout << presenter.reportMeans() << '\n';
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }
    slog::info << "Execution successful" << slog::endl;
    return 0;
}
