// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include "person_action_detector.hpp"
#if defined(HAVE_OPENCV_GAPI)
#include <samples/ocv_common.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

std::string getBinPath(const std::string & pathXML) {
    std::string pathBIN(pathXML); 
    return pathBIN.replace(pathBIN.size() - 3, 3 , "bin");
}
namespace show {
    const std::string actions[] = {
        "sitting", "standing", "rising hand"
    };

    void DrawResults(cv::Mat &frame, const std::vector<cv::Rect> &faces) {
        for (auto it = faces.begin(); it != faces.end(); ++it) {
            const auto &rc = *it;
            cv::rectangle(frame, rc, {200, 200, 200},  2);
        }
    }

    void DrawResults(cv::Mat &frame, const std::vector<Detections> &persons) {
        for (auto it = persons.begin(); it != persons.end(); ++it) {
            const auto &rc = (*it).rect;
            cv::rectangle(frame, rc, {200, 200, 200},  2);
            std::stringstream ss;
            ss << actions[(*it).action_label];
            cv::putText(frame, ss.str(),
                        cv::Point(rc.x, rc.y - 15),
                        cv::FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        cv::Scalar(0, 0, 255));
        }
    }
} // namespace show

namespace {
    const std::string about =
        "classroom";    
    const std::string keys =
    "{ h help |   | print this help message }"
    "{ input  |   | Path to an input video file }"
    "{ fdm    |   | IE face detection model IR }"
    "{ pam    |   | IE person detection action recognition model IR }" 
    "{ fdd    |   | IE face detection device }";
}

namespace custom {
    G_API_NET(Faces, <cv::GMat(cv::GMat)>, "face-detector");

    using PAInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat>;
    G_API_NET(PersAction, <PAInfo(cv::GMat)>, "person-detection-action-recognition");


    G_API_OP(PostProc, <cv::GArray<cv::Rect>(cv::GMat, cv::GMat)>, "custom.fd_postproc") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };
    G_API_OP(PersonDetActionRec, <cv::GArray<Detections>(cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat, cv::GMat)>,
                                                 "custom.person_detection_action_recognition_postproc") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &,
                                      const cv::GMatDesc &, const cv::GMatDesc &,
                                      const cv::GMatDesc &, const cv::GMatDesc &,
                                      const cv::GMatDesc &, const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };
    GAPI_OCV_KERNEL(OCVPostProc, PostProc) {
        static void run(const cv::Mat &in_ssd_result,
                        const cv::Mat &in_frame,
                        std::vector<cv::Rect> &out_faces) {
            const int MAX_PROPOSALS = 200;
            const int OBJECT_SIZE   =   7;
            const cv::Size upscale = in_frame.size();
            const cv::Rect surface({0,0}, upscale);

            out_faces.clear();

            const float *data = in_ssd_result.ptr<float>();
            for (int i = 0; i < MAX_PROPOSALS; i++) {
                const float image_id   = data[i * OBJECT_SIZE + 0]; 
                const float confidence = data[i * OBJECT_SIZE + 2];
                const float rc_left    = data[i * OBJECT_SIZE + 3];
                const float rc_top     = data[i * OBJECT_SIZE + 4];
                const float rc_right   = data[i * OBJECT_SIZE + 5];
                const float rc_bottom  = data[i * OBJECT_SIZE + 6];
                if (image_id < 0.f) { 
                    break;
                }
                if (confidence < 0.25f) { 
                    continue;
                }

                cv::Rect rc;
                rc.x      = static_cast<int>(rc_left   * upscale.width);
                rc.y      = static_cast<int>(rc_top    * upscale.height);
                rc.width  = static_cast<int>(rc_right  * upscale.width)  - rc.x;
                rc.height = static_cast<int>(rc_bottom * upscale.height) - rc.y;
                out_faces.push_back(rc & surface);
            }
        }
    };
    GAPI_OCV_KERNEL(OCVPersonDetActionRec, PersonDetActionRec) {
        static void run(const cv::Mat &in_ssd_local,
                        const cv::Mat &in_ssd_conf,
                        const cv::Mat &in_ssd_priorbox,
                        const cv::Mat &in_ssd_anchor1,
                        const cv::Mat &in_ssd_anchor2,
                        const cv::Mat &in_ssd_anchor3,
                        const cv::Mat &in_ssd_anchor4,
                        const cv::Mat &in_frame, std::vector<Detections> &out_detections) {
            const float *local_data    = reinterpret_cast<float*>(in_ssd_local.data);
            const float *det_conf_data = reinterpret_cast<float*>(in_ssd_conf.data);
            const float *prior_data    = reinterpret_cast<float*>(in_ssd_priorbox.data);

            std::vector<float*> action_conf_data;
            action_conf_data.push_back(reinterpret_cast<float*>(in_ssd_anchor1.data));
            action_conf_data.push_back(reinterpret_cast<float*>(in_ssd_anchor2.data));
            action_conf_data.push_back(reinterpret_cast<float*>(in_ssd_anchor3.data));
            action_conf_data.push_back(reinterpret_cast<float*>(in_ssd_anchor4.data));
                
            const cv::Size frame_size = in_frame.size();
            std::unique_ptr<ActionDetector> detector(new ActionDetector);
            detector->GetPostProcResult(local_data, det_conf_data, prior_data, action_conf_data, frame_size, out_detections);
        }
    };
}// namespace custom
    
int main(int argc, char* argv[]) {
    cv::CommandLineParser cmd(argc, argv, keys);
    if (cmd.has("help")) {
        cmd.printMessage();
        return 0;
    }

    cv::GComputation pp([]() {
        cv::GMat in;
        cv::GMat detections = cv::gapi::infer<custom::Faces>(in);
        cv::GArray<cv::Rect> faces = custom::PostProc::on(detections, in);

        cv::GMat location;
        cv::GMat detect_confidences;
        cv::GMat priorboxes;
        cv::GMat action_con1;
        cv::GMat action_con2;
        cv::GMat action_con3;
        cv::GMat action_con4;

        std::tie(location, detect_confidences, priorboxes, 
                 action_con1, action_con2, action_con3, action_con4) = cv::gapi::infer<custom::PersAction>(in);

        cv::GArray<Detections> persons = custom::PersonDetActionRec::on(location, detect_confidences, priorboxes,
                                                                action_con1, action_con2, action_con3, action_con4,
                                                                in);

        cv::GMat frame = cv::gapi::copy(in);
        return cv::GComputation(cv::GIn(in), cv::GOut(frame, faces, persons));
    });


    std::string fdPathXML = cmd.get<std::string>("fdm"); 
    auto det_net = cv::gapi::ie::Params<custom::Faces> {
        fdPathXML,
        getBinPath(fdPathXML),
        cmd.get<std::string>("fdd"),
    };

    std::string paPathXML = cmd.get<std::string>("pam"); 
    auto pos_act_net = cv::gapi::ie::Params<custom::PersAction> {
        paPathXML,
        getBinPath(paPathXML),
        cmd.get<std::string>("fdd"),
    }.cfgOutputLayers({"mbox_loc1/out/conv/flat", 
                       "mbox_main_conf/out/conv/flat/softmax/flat",
                       "mbox/priorbox",
                       "out/anchor1",
                       "out/anchor2",
                       "out/anchor3",
                       "out/anchor4"
                       }); // 0006 have another outputs
    auto kernels = cv::gapi::kernels<custom::OCVPostProc, custom::OCVPersonDetActionRec>();
    auto networks = cv::gapi::networks(det_net, pos_act_net);

    auto cc = pp.compileStreaming(cv::compile_args(kernels, networks));

    const std::string input = cmd.get<std::string>("input");
    auto in_src = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(input);
    cc.setSource(cv::gin(in_src));
    cc.start();
    

    cv::Mat frame;
    std::vector<cv::Rect> faces;
    std::vector<Detections> persons;
    while (cc.running()) {
        auto out_vector = cv::gout(frame, faces, persons);
        if (!cc.try_pull(std::move(out_vector))) {
            if (cv::waitKey(1) >= 0) break;
            else continue;
        }
        show::DrawResults(frame, faces);
        show::DrawResults(frame, persons);
        cv::imshow("Result", frame);
    }
	std::cout << "==========OK==========" << std::endl;
    return 0;
}
#endif  // HAVE_OPECV_GAPI
