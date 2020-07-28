// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>  // NOLINT

#include <gflags/gflags.h>
#include <monitors/presenter.h>
#include <samples/ocv_common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <deque>
#include <map>
#include <set>
#include <algorithm>
#include <utility>
#include <ie_iextension.h>

#include "actions.hpp"
#include "action_detector.hpp"
#include "detector.hpp"
#include "face_reid.hpp"
#include "tracker.hpp"
#include "logger.hpp"
#include "smart_classroom_demo.hpp"

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

using namespace InferenceEngine;

namespace {

class Visualizer {
private:
    cv::Mat frame_;
    cv::Mat top_persons_;
    const bool enabled_;
    const int num_top_persons_;
    cv::VideoWriter& writer_;
    float rect_scale_x_;
    float rect_scale_y_;
    static int const max_input_width_ = 1920;
    std::string const main_window_name_ = "Smart classroom demo";
    std::string const top_window_name_ = "Top-k students";
    static int const crop_width_ = 128;
    static int const crop_height_ = 320;
    static int const header_size_ = 80;
    static int const margin_size_ = 5;

public:
    Visualizer(bool enabled, cv::VideoWriter& writer, int num_top_persons) : enabled_(enabled), num_top_persons_(num_top_persons), writer_(writer),
                                                        rect_scale_x_(0), rect_scale_y_(0) {
        if (!enabled_) {
            return;
        }

        cv::namedWindow(main_window_name_);

        if (num_top_persons_ > 0) {
            cv::namedWindow(top_window_name_);

            CreateTopWindow();
            ClearTopWindow();
        }
    }

    static cv::Size GetOutputSize(const cv::Size& input_size) {
        if (input_size.width > max_input_width_) {
            float ratio = static_cast<float>(input_size.height) / input_size.width;
            return cv::Size(max_input_width_, cvRound(ratio*max_input_width_));
        }
        return input_size;
    }

    void SetFrame(const cv::Mat& frame) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        frame_ = frame.clone();
        rect_scale_x_ = 1;
        rect_scale_y_ = 1;
        cv::Size new_size = GetOutputSize(frame_.size());
        if (new_size != frame_.size()) {
            rect_scale_x_ = static_cast<float>(new_size.height) / frame_.size().height;
            rect_scale_y_ = static_cast<float>(new_size.width) / frame_.size().width;
            cv::resize(frame_, frame_, new_size);
        }
    }

    void Show() const {
        if (enabled_) {
            cv::imshow(main_window_name_, frame_);
        }

        if (writer_.isOpened()) {
            writer_ << frame_;
        }
    }

    void DrawCrop(cv::Rect roi, int id, const cv::Scalar& color) const {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        if (id < 0 || id >= num_top_persons_) {
            return;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
            roi.x = cvRound(roi.x * rect_scale_x_);
            roi.y = cvRound(roi.y * rect_scale_y_);

            roi.height = cvRound(roi.height * rect_scale_y_);
            roi.width = cvRound(roi.width * rect_scale_x_);
        }

        roi.x = std::max(0, roi.x);
        roi.y = std::max(0, roi.y);
        roi.width = std::min(roi.width, frame_.cols - roi.x);
        roi.height = std::min(roi.height, frame_.rows - roi.y);

        const auto crop_label = std::to_string(id + 1);

        auto frame_crop = frame_(roi).clone();
        cv::resize(frame_crop, frame_crop, cv::Size(crop_width_, crop_height_));

        const int shift = (id + 1) * margin_size_ + id * crop_width_;
        frame_crop.copyTo(top_persons_(cv::Rect(shift, header_size_, crop_width_, crop_height_)));

        cv::imshow(top_window_name_, top_persons_);
    }

    void DrawObject(cv::Rect rect, const std::string& label_to_draw,
                    const cv::Scalar& text_color, const cv::Scalar& bbox_color, bool plot_bg) {
        if (!enabled_ && !writer_.isOpened()) {
            return;
        }

        if (rect_scale_x_ != 1 || rect_scale_y_ != 1) {
            rect.x = cvRound(rect.x * rect_scale_x_);
            rect.y = cvRound(rect.y * rect_scale_y_);

            rect.height = cvRound(rect.height * rect_scale_y_);
            rect.width = cvRound(rect.width * rect_scale_x_);
        }
        cv::rectangle(frame_, rect, bbox_color);

        if (plot_bg && !label_to_draw.empty()) {
            int baseLine = 0;
            const cv::Size label_size =
                cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            cv::rectangle(frame_, cv::Point(rect.x, rect.y - label_size.height),
                            cv::Point(rect.x + label_size.width, rect.y + baseLine),
                            bbox_color, cv::FILLED);
        }
        if (!label_to_draw.empty()) {
            cv::putText(frame_, label_to_draw, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_PLAIN, 1,
                        text_color, 1, cv::LINE_AA);
        }
    }

    void DrawFPS(const float fps, const cv::Scalar& color) {
        if (enabled_ && !writer_.isOpened()) {
            cv::putText(frame_,
                        std::to_string(static_cast<int>(fps)) + " fps",
                        cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                        color, 2, cv::LINE_AA);
        }
    }

    void CreateTopWindow() {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        const int width = margin_size_ * (num_top_persons_ + 1) + crop_width_ * num_top_persons_;
        const int height = header_size_ + crop_height_ + margin_size_;

        top_persons_.create(height, width, CV_8UC3);
    }

    void ClearTopWindow() {
        if (!enabled_ || num_top_persons_ <= 0) {
            return;
        }

        top_persons_.setTo(cv::Scalar(255, 255, 255));

        for (int i = 0; i < num_top_persons_; ++i) {
            const int shift = (i + 1) * margin_size_ + i * crop_width_;

            cv::rectangle(top_persons_, cv::Point(shift, header_size_),
                          cv::Point(shift + crop_width_, header_size_ + crop_height_),
                          cv::Scalar(128, 128, 128), cv::FILLED);

            const auto label_to_draw = "#" + std::to_string(i + 1);
            int baseLine = 0;
            const auto label_size =
                cv::getTextSize(label_to_draw, cv::FONT_HERSHEY_SIMPLEX, 2, 2, &baseLine);
            const int text_shift = (crop_width_ - label_size.width) / 2;
            cv::putText(top_persons_, label_to_draw,
                        cv::Point(shift + text_shift, label_size.height + baseLine / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        cv::imshow(top_window_name_, top_persons_);
    }

    void Finalize() const {
        if (enabled_) {
            cv::destroyWindow(main_window_name_);

            if (num_top_persons_ > 0) {
                cv::destroyWindow(top_window_name_);
            }
        }

        if (writer_.isOpened()) {
            writer_.release();
        }
    }
};

const int default_action_index = -1;  // Unknown action class

void ConvertActionMapsToFrameEventTracks(const std::vector<std::map<int, int>>& obj_id_to_action_maps,
                                         int default_action,
                                         std::map<int, FrameEventsTrack>* obj_id_to_actions_track) {
    for (size_t frame_id = 0; frame_id < obj_id_to_action_maps.size(); ++frame_id) {
        for (const auto& tup : obj_id_to_action_maps[frame_id]) {
            if (tup.second != default_action) {
                (*obj_id_to_actions_track)[tup.first].emplace_back(frame_id, tup.second);
            }
        }
    }
}

void SmoothTracks(const std::map<int, FrameEventsTrack>& obj_id_to_actions_track,
                  int start_frame, int end_frame, int window_size, int min_length, int default_action,
                  std::map<int, RangeEventsTrack>* obj_id_to_events) {
    // Iterate over face tracks
    for (const auto& tup : obj_id_to_actions_track) {
        const auto& frame_events = tup.second;
        if (frame_events.empty()) {
            continue;
        }

        RangeEventsTrack range_events;


        // Merge neighbouring events and filter short ones
        range_events.emplace_back(frame_events.front().frame_id,
                                  frame_events.front().frame_id + 1,
                                  frame_events.front().action);

        for (size_t frame_id = 1; frame_id < frame_events.size(); ++frame_id) {
            const auto& last_range_event = range_events.back();
            const auto& cur_frame_event = frame_events[frame_id];

            if (last_range_event.end_frame_id + window_size - 1 >= cur_frame_event.frame_id &&
                last_range_event.action == cur_frame_event.action) {
                range_events.back().end_frame_id = cur_frame_event.frame_id + 1;
            } else {
                if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
                    range_events.pop_back();
                }

                range_events.emplace_back(cur_frame_event.frame_id,
                                          cur_frame_event.frame_id + 1,
                                          cur_frame_event.action);
            }
        }
        if (range_events.back().end_frame_id - range_events.back().begin_frame_id < min_length) {
            range_events.pop_back();
        }

        // Extrapolate track
        if (range_events.empty()) {
            range_events.emplace_back(start_frame, end_frame, default_action);
        } else {
            range_events.front().begin_frame_id = start_frame;
            range_events.back().end_frame_id = end_frame;
        }

        // Interpolate track
        for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
            auto& last_event = range_events[event_id - 1];
            auto& cur_event = range_events[event_id];

            int middle_point = static_cast<int>(0.5f * (cur_event.begin_frame_id + last_event.end_frame_id));

            cur_event.begin_frame_id = middle_point;
            last_event.end_frame_id = middle_point;
        }

        // Merge consecutive events
        auto& final_events = (*obj_id_to_events)[tup.first];
        final_events.push_back(range_events.front());
        for (size_t event_id = 1; event_id < range_events.size(); ++event_id) {
            const auto& cur_event = range_events[event_id];

            if (final_events.back().action == cur_event.action) {
                final_events.back().end_frame_id = cur_event.end_frame_id;
            } else {
                final_events.push_back(cur_event);
            }
        }
    }
}

void ConvertRangeEventsTracksToActionMaps(int num_frames,
                                          const std::map<int, RangeEventsTrack>& obj_id_to_events,
                                          std::vector<std::map<int, int>>* obj_id_to_action_maps) {
    obj_id_to_action_maps->resize(num_frames);

    for (const auto& tup : obj_id_to_events) {
        const int obj_id = tup.first;
        const auto& events = tup.second;

        for (const auto& event : events) {
            for (int frame_id = event.begin_frame_id; frame_id < event.end_frame_id; ++frame_id) {
                (*obj_id_to_action_maps)[frame_id].emplace(obj_id, event.action);
            }
        }
    }
}

std::vector<std::string> ParseActionLabels(const std::string& in_str) {
    std::vector<std::string> labels;
    std::string label;
    std::istringstream stream_to_split(in_str);

    while (std::getline(stream_to_split, label, ',')) {
      labels.push_back(label);
    }

    return labels;
}

std::string GetActionTextLabel(const unsigned label, const std::vector<std::string>& actions_map) {
    if (label < actions_map.size()) {
        return actions_map[label];
    }
    return "__undefined__";
}

cv::Scalar GetActionTextColor(const unsigned label) {
    static const cv::Scalar label_colors[] = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255)};
    if (label < arraySize(label_colors)) {
        return label_colors[label];
    }
    return cv::Scalar(0, 0, 0);
}

float CalculateIoM(const cv::Rect& rect1, const cv::Rect& rect2) {
  int area1 = rect1.area();
  int area2 = rect2.area();

  float area_min = static_cast<float>(std::min(area1, area2));
  float area_intersect = static_cast<float>((rect1 & rect2).area());

  return area_intersect / area_min;
}

cv::Rect DecreaseRectByRelBorders(const cv::Rect& r) {
    float w = static_cast<float>(r.width);
    float h = static_cast<float>(r.height);

    float left = std::ceil(w * 0.0f);
    float top = std::ceil(h * 0.0f);
    float right = std::ceil(w * 0.0f);
    float bottom = std::ceil(h * .7f);

    cv::Rect res;
    res.x = r.x + static_cast<int>(left);
    res.y = r.y + static_cast<int>(top);
    res.width = static_cast<int>(r.width - left - right);
    res.height = static_cast<int>(r.height - top - bottom);
    return res;
}

int GetIndexOfTheNearestPerson(const TrackedObject& face, const std::vector<TrackedObject>& tracked_persons) {
    int argmax = -1;
    float max_iom = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < tracked_persons.size(); i++) {
        float iom = CalculateIoM(face.rect, DecreaseRectByRelBorders(tracked_persons[i].rect));
        if ((iom > 0) && (iom > max_iom)) {
            max_iom = iom;
            argmax = i;
        }
    }
    return argmax;
}

std::map<int, int> GetMapFaceTrackIdToLabel(const std::vector<Track>& face_tracks) {
    std::map<int, int> face_track_id_to_label;
    for (const auto& track : face_tracks) {
        const auto& first_obj = track.first_object;
        // check consistency
        // to receive this consistency for labels
        // use the function UpdateTrackLabelsToBestAndFilterOutUnknowns
        for (const auto& obj : track.objects) {
            SCR_CHECK_EQ(obj.label, first_obj.label);
            SCR_CHECK_EQ(obj.object_id, first_obj.object_id);
        }

        auto cur_obj_id = first_obj.object_id;
        auto cur_label = first_obj.label;
        SCR_CHECK(face_track_id_to_label.count(cur_obj_id) == 0) << " Repeating face tracks";
        face_track_id_to_label[cur_obj_id] = cur_label;
    }
    return face_track_id_to_label;
}

// bool checkDynamicBatchSupport(const Core& ie, const std::string& device)  {
//     try  {
//         if (ie.GetConfig(device, CONFIG_KEY(DYN_BATCH_ENABLED)).as<std::string>() != PluginConfigParams::YES)
//             return false;
//     }
//     catch(const std::exception&)  {
//         return false;
//     }
//     return true;
// }
struct FaceRecognizerConfig {
    double reid_threshold;
    std::vector<GalleryObject> identities;
    std::vector<int> idx_to_id;
    bool greedy_reid_matching;
    bool reid_realizable = false;
    double actions_type;
};

class FaceRecognizer {
public:
    virtual ~FaceRecognizer() = default;

    virtual bool LabelExists(const std::string &label) const = 0;
    virtual std::string GetLabelByID(int id) const = 0;
    virtual std::vector<std::string> GetIDToLabelMap() const = 0;

    virtual std::vector<int> Recognize(const std::vector<cv::Mat> &,
                                       const detection::DetectedObjects &) = 0;
};

class FaceRecognizerNull : public FaceRecognizer {
public:
    bool LabelExists(const std::string &) const override { return false; }

    std::string GetLabelByID(int) const override {
        return EmbeddingsGallery::unknown_label;
    }

    std::vector<std::string> GetIDToLabelMap() const override { return {}; }

    std::vector<int> Recognize(const std::vector<cv::Mat>& embeddings,
                               const detection::DetectedObjects& faces) override {
        return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
    }
};

class FaceRecognizerDefault : public FaceRecognizer {
public:
    FaceRecognizerDefault(FaceRecognizerConfig config)
        : face_gallery(config.reid_threshold,
                       config.identities,
                       config.idx_to_id,
                       config.greedy_reid_matching) {}

    bool LabelExists(const std::string &label) const override {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const override {
        return face_gallery.GetLabelByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const override {
        return face_gallery.GetIDToLabelMap();
    }

    std::vector<int> Recognize(const std::vector<cv::Mat>& embeddings,
                               const detection::DetectedObjects& faces) override {
        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

private:    
    EmbeddingsGallery face_gallery;
};

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------

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

} // namespace

struct TrackerParamsPack {
    TrackerParams tracker_reid_params;
    TrackerParams tracker_action_params;
};

struct TrackersPack {    
    TrackersPack( TrackerParams tracker_reid_params
                , TrackerParams tracker_action_params) : tracker_reid(Tracker(tracker_reid_params))
                                                       , tracker_action(Tracker(tracker_action_params)) {}
    Tracker tracker_reid;
    Tracker tracker_action;
};

namespace config {
    detection::DetectorConfig face_config;
    detection::DetectorConfig face_registration_det_config;
    ActionDetectorConfig action_config;
    FaceRecognizerConfig rec_config;
} // namespace config
namespace state {
    /** NOTE: Questionable solution
      * kernels use num_frames,
      * num_frames total state need for FPS counter
      */
    size_t work_num_frames = 0;
    size_t total_num_frames = 0;
} // namespace state
namespace util {
    std::string GetBinPath(const std::string &pathXML) {
        CV_Assert(pathXML.substr(pathXML.size() - 4, pathXML.size()) == ".xml");
        std::string pathBIN(pathXML);
        return pathBIN.replace(pathBIN.size() - 3, 3, "bin");
    }
    bool IsNetworkNewVersion(const std::string &model_path) {
        CV_Assert(!model_path.empty());
        return model_path.at(model_path.size() - 5) == '6';
    }
    struct Avg {
        struct Elapsed {
            explicit Elapsed(double ms) : ss(ms / 1000.),
                mm(static_cast<int>(std::lround(ss / 60))) {}
            const double ss;
            const int    mm;
        };

        using MS = std::chrono::duration<double, std::ratio<1, 1000>>;
        using TS = std::chrono::time_point<std::chrono::high_resolution_clock>;
        TS started;

        void    start() { started = now(); }
        TS      now() const { return std::chrono::high_resolution_clock::now(); }
        double  tick() const { return std::chrono::duration_cast<MS>(now() - started).count(); }
        Elapsed elapsed() const { return Elapsed{ tick() }; }
        double  fps(std::size_t n) const { return static_cast<double>(n) / (tick() / 1000.); }
    };
} // namespace util
namespace cv
{
    namespace detail
    {
        template<> struct CompileArgTag<TrackerParamsPack>
        {
            static const char* tag()
            {
                return "custom.get_recognize_result_state_params";
            }
        };
        template<> struct CompileArgTag<TrackerParams>
        {
            static const char* tag()
            {
                return "custom.get_action_detection_result_state_params";
            }
        };
    }
}

namespace custom {
    /** NETS */
    G_API_NET( FaceDetector, <cv::GMat(cv::GMat)>, "face-detector");
    G_API_NET( LandmarksDetector, <cv::GMat(cv::GMat)>, "landmarks-detector");
    G_API_NET( FaceReidentificator, <cv::GMat(cv::GMat)>
             , "face-reidentificator");
    using PAInfo = std::tuple<cv::GMat, cv::GMat, cv::GMat, cv::GMat,
                              cv::GMat, cv::GMat, cv::GMat>;
    G_API_NET( PersonDetActionRec, <PAInfo(cv::GMat)>
             , "person-detection-action-recognition");
    /** OPs */
    G_API_OP(FaceDetectorPostProc,
             <cv::GArray<detection::DetectedObject>(cv::GMat,
                                                    cv::GMat,
                                                    detection::DetectorConfig)>,
             "custom.fd_postproc") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &,
                                      const cv::GMatDesc &,
                                      const detection::DetectorConfig &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP(GetRectFromImage,
             <cv::GArray<cv::Rect>(cv::GMat)>,
             "custom.get_rect_from_image") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP(GetRectsFromDetections,
             <cv::GArray<cv::Rect>(cv::GArray<detection::DetectedObject>)>,
             "custom.get_rects_from_detection") {
        static cv::GArrayDesc outMeta(const cv::GArrayDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( PersonDetActionRecPostProc
            , <cv::GArray<DetectedAction>(cv::GMat, cv::GMat,
                                          cv::GMat, cv::GMat,
                                          cv::GMat, cv::GMat,
                                          cv::GMat, cv::GMat,
                                          ActionDetectorConfig)>,
             "custom.person_detection_action_recognition_postproc") {
        static cv::GArrayDesc outMeta(const cv::GMatDesc &, const cv::GMatDesc &,
                                      const cv::GMatDesc &, const cv::GMatDesc &,
                                      const cv::GMatDesc &, const cv::GMatDesc &,
                                      const cv::GMatDesc &, const cv::GMatDesc &,
                                      const ActionDetectorConfig &) {
            return cv::empty_array_desc();
        }
    };

    using RECInfo = std::tuple<cv::GArray<TrackedObject>,
                               cv::GArray<TrackedObject>,
                               cv::GArray<std::string>,
                               cv::GArray<std::string>,
                               cv::GArray<Track>>;

    G_API_OP( GetRecognizeResult
            , <RECInfo( cv::GArray<cv::GMat>
                      , cv::GArray<detection::DetectedObject>
                      , cv::GArray<DetectedAction>
                      , cv::GMat
                      , FaceRecognizerConfig)>
            , "custom.get_recognize_result") {
        static std::tuple < cv::GArrayDesc
                          , cv::GArrayDesc
                          , cv::GArrayDesc
                          , cv::GArrayDesc
                          , cv::GArrayDesc> outMeta( const cv::GArrayDesc &
                                                   , const cv::GArrayDesc &
                                                   , const cv::GArrayDesc &
                                                   , const cv::GMatDesc &
                                                   , const FaceRecognizerConfig &) {
            return std::make_tuple( cv::empty_array_desc()
                                  , cv::empty_array_desc()
                                  , cv::empty_array_desc()
                                  , cv::empty_array_desc()
                                  , cv::empty_array_desc());
        }
    };

    G_API_OP( GetActionTopHandsDetectionResult
            , <cv::GArray<TrackedObject>( cv::GArray<DetectedAction>
                                        , cv::GMat)>
            , "custom.get_action_detection_result_for_top_k_first_hands") {
        static cv::GArrayDesc outMeta( const cv::GArrayDesc &
                                     , const cv::GMatDesc &) {
            return cv::empty_array_desc();
        }
    };

    G_API_OP( AlignFacesForReidentification
            , <cv::GMat( cv::GArray<cv::GMat>
                       , cv::GArray<cv::Rect>
                       , cv::GMat)>
            , "custom.align_faces_for_reidentification") {
        static cv::GMatDesc outMeta(const cv::GArrayDesc &,
            const cv::GArrayDesc &,
            const cv::GMatDesc & in) {
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

    /** OCV KERNELs */
    GAPI_OCV_KERNEL(OCVFaceDetectorPostProc, FaceDetectorPostProc) {
        static void run(const cv::Mat &in_ssd_result,
                        const cv::Mat &in_frame,
                        const detection::DetectorConfig & face_config,
                        std::vector<detection::DetectedObject> &out_faces) {
            std::unique_ptr<detection::FaceDetection> face_detector(new detection::FaceDetection(face_config));
            out_faces = face_detector->fetchResults(in_ssd_result, in_frame);
        }
    };

    GAPI_OCV_KERNEL(OCVGetRectFromImage, GetRectFromImage) {
        static void run(const cv::Mat &in_image,
                        std::vector<cv::Rect> &out_rects) {
            out_rects.emplace_back(cv::Rect(0, 0, in_image.cols, in_image.rows));
        }
    };

    GAPI_OCV_KERNEL(OCVGetRectsFromDetections, GetRectsFromDetections) {
        static void run(const detection::DetectedObjects &detections,
                        std::vector<cv::Rect> &out_rects) {
            for (const auto& it : detections) {
                out_rects.emplace_back(it.rect);
            }
        }
    };

    GAPI_OCV_KERNEL_ST(OCVGetRecognizeResult,  GetRecognizeResult, TrackersPack) {
        static void setup(const cv::GArrayDesc &,
                          const cv::GArrayDesc &,
                          const cv::GArrayDesc &,
                          const cv::GMatDesc &,
                          const FaceRecognizerConfig &,
                          std::shared_ptr<TrackersPack> &trackers,
                          const cv::GCompileArgs &compileArgs) {
            auto trParamsPack = cv::gapi::getCompileArg<TrackerParamsPack>(compileArgs)
                 .value_or(TrackerParamsPack{});
            trackers = std::make_shared<TrackersPack>(trParamsPack.tracker_reid_params,                                
                                                      trParamsPack.tracker_action_params);
        }
        static void run(const std::vector<cv::Mat>& embeddings,
                        const detection::DetectedObjects &faces,
                        const DetectedActions & actions,
                        const cv::Mat &frame,
                        const FaceRecognizerConfig &rec_config,
                        TrackedObjects &tracked_faces,
                        TrackedObjects &tracked_actions,
                        std::vector<std::string> &face_labels,
                        std::vector<std::string> &face_id_to_label_map,
                        std::vector<Track> &face_tracks,
                        TrackersPack &trackers) {
            // NOTE: Recognize works with shape 256:1 after G-API we get 1:256:1:1
            std::vector<cv::Mat> reshape_emb;
            for (auto & emb : embeddings) {
                reshape_emb.emplace_back(emb.reshape(1, { emb.size().width, 1 }));
            }            
            std::unique_ptr<FaceRecognizer> face_recognizer;

            if (rec_config.reid_realizable) {
                face_recognizer.reset(new FaceRecognizerDefault(rec_config));
            } else {
                face_recognizer.reset(new FaceRecognizerNull);
            }
            std::vector<int> ids = face_recognizer->Recognize(reshape_emb, faces);
            TrackedObjects tracked_face_objects;
            for (size_t i = 0; i < faces.size(); i++) {
                tracked_face_objects.emplace_back(faces[i].rect, faces[i].confidence, ids[i]);
            }
            trackers.tracker_reid.Process(frame, tracked_face_objects, state::work_num_frames);

            tracked_faces = trackers.tracker_reid.TrackedDetectionsWithLabels();

            for (size_t j = 0; j < tracked_faces.size(); j++) {
                const auto& face = tracked_faces[j];
                face_labels.push_back(face_recognizer->GetLabelByID(face.label));
            }
            TrackedObjects tracked_action_objects;
            for (const auto& action : actions) {
                tracked_action_objects.emplace_back(action.rect, action.detection_conf, action.label);
            }
            
            trackers.tracker_action.Process(frame, tracked_action_objects, state::work_num_frames);
            tracked_actions = trackers.tracker_action.TrackedDetectionsWithLabels();

            if (rec_config.actions_type == STUDENT) {
                face_tracks = trackers.tracker_reid.vector_tracks();
                face_id_to_label_map = face_recognizer->GetIDToLabelMap();
            }
        }
    };

    GAPI_OCV_KERNEL_ST( OCVGetActionTopHandsDetectionResult
                      , GetActionTopHandsDetectionResult
                      , Tracker) {
        static void setup( const cv::GArrayDesc &
                         , const cv::GMatDesc &
                         , std::shared_ptr<Tracker> &tracker_action
                         , const cv::GCompileArgs &compileArgs) {
            auto trParam = cv::gapi::getCompileArg<TrackerParams>(compileArgs)
                .value_or(TrackerParams{});
            tracker_action = std::make_shared<Tracker>(trParam);
        }
        static void run( const DetectedActions & actions
                       , const cv::Mat &frame
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

    GAPI_OCV_KERNEL( OCVAlignFacesForReidentification
                   , AlignFacesForReidentification) {
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
                   , PersonDetActionRecPostProc) {
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
            out_detections = action_detector->fetchResults( in_ssd_local
                                                          , in_ssd_conf
                                                          , in_ssd_priorbox
                                                          , in_ssd_anchor1
                                                          , in_ssd_anchor2
                                                          , in_ssd_anchor3
                                                          , in_ssd_anchor4
                                                          , in_frame);
            
        }
    };

    /** NOTE: Three OCV KERNELs for empty {} GArray **/
    GAPI_OCV_KERNEL(OCVGetEmptyFaces, GetEmptyFaces) {
        static void run(const cv::Mat &in,
                        std::vector<detection::DetectedObject> &empty_faces) {
            empty_faces = {};
        }
    };

    GAPI_OCV_KERNEL(OCVGetEmptyActions, GetEmptyActions) {
        static void run(const cv::Mat &in,
                        std::vector<DetectedAction> &empty_actions) {
            empty_actions = {};
        }
    };

    GAPI_OCV_KERNEL(OCVGetEmptyMatGArray, GetEmptyMatGArray) {
        static void run(const cv::Mat &in,
                        std::vector<cv::Mat> &empty_garray) {
            empty_garray = {};
        }
    };
} // namespace custom

namespace {
    bool file_exists(const std::string& name) {
        std::ifstream f(name.c_str());
        return f.good();
    }

    inline char separator() {
#ifdef _WIN32
        return '\\';
#else
        return '/';
#endif
    }

    std::string folder_name(const std::string& path) {
        size_t found_pos;
        found_pos = path.find_last_of(separator());
        if (found_pos != std::string::npos)
            return path.substr(0, found_pos);
        return std::string(".") + separator();
    }
}  // namespace

int main(int argc, char* argv[]) {
    try {
        /** This demo covers 4 certain topologies and cannot be generalized **/
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        const auto video_path = FLAGS_i;
        const auto ad_model_path = FLAGS_m_act;
        const auto fd_model_path = FLAGS_m_fd;
        const auto fr_model_path = FLAGS_m_reid;
        const auto lm_model_path = FLAGS_m_lm;
        const auto teacher_id = FLAGS_teacher_id;
        const auto ids_list = FLAGS_fg;

        if (!FLAGS_teacher_id.empty() && !FLAGS_top_id.empty()) {
            slog::err << "Cannot run simultaneously teacher action and top-k students recognition."
                      << slog::endl;
            return 1;
        }

        const auto actions_type = FLAGS_teacher_id.empty()
                                      ? FLAGS_a_top > 0 ? TOP_K : STUDENT
                                      : TEACHER;
        const auto actions_map = actions_type == STUDENT
                                     ? ParseActionLabels(FLAGS_student_ac)
                                     : actions_type == TOP_K
                                         ? ParseActionLabels(FLAGS_top_ac)
                                         : ParseActionLabels(FLAGS_teacher_ac);
        const auto num_top_persons = actions_type == TOP_K ? FLAGS_a_top : -1;
        const auto top_action_id = actions_type == TOP_K
                                   ? std::distance(actions_map.begin(), find(actions_map.begin(), actions_map.end(), FLAGS_top_id))
                                   : -1;
        if (actions_type == TOP_K && (top_action_id < 0 || top_action_id >= static_cast<int>(actions_map.size()))) {
            slog::err << "Cannot find target action: " << FLAGS_top_id << slog::endl;
            return 1;
        }

        slog::info << "Reading video '" << video_path << "'" << slog::endl;
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        std::vector<std::string> devices = {FLAGS_d_act, FLAGS_d_fd, FLAGS_d_lm,
                                            FLAGS_d_reid};
        std::set<std::string> loadedDevices;

        slog::info << "Device info: " << slog::endl;

        for (const auto &device : devices) {
            if (loadedDevices.find(device) != loadedDevices.end())
                continue;

            std::cout << ie.GetVersions(device) << std::endl;

            /** Load extensions for the CPU device **/
            if ((device.find("CPU") != std::string::npos)) {
                if (!FLAGS_l.empty()) {
                    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
                    auto extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
                    ie.AddExtension(extension_ptr, "CPU");
                    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
                }
            } else if (!FLAGS_c.empty()) {
                // Load Extensions for other plugins not CPU
                ie.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}}, "GPU");
            }

            if (device.find("CPU") != std::string::npos) {
                ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "CPU");
            } else if (device.find("GPU") != std::string::npos) {
                ie.SetConfig({{PluginConfigParams::KEY_DYN_BATCH_ENABLED, PluginConfigParams::YES}}, "GPU");
            }

            loadedDevices.insert(device);
        }

        // NOTE: Empty strings instead empty object of Params class 
        cv::gapi::ie::Params<custom::FaceDetector> det_net("","","");
        cv::gapi::ie::Params<custom::LandmarksDetector> landm_net("","","");
        cv::gapi::ie::Params<custom::FaceReidentificator> reident_net("","","");
        cv::gapi::ie::Params<custom::PersonDetActionRec> action_net("", "", "");

        if (!ad_model_path.empty()) {
            // Load action detector
            config::action_config.new_network = util::IsNetworkNewVersion(ad_model_path);
            config::action_config.detection_confidence_threshold = static_cast<float>(FLAGS_t_ad);
            config::action_config.action_confidence_threshold = static_cast<float>(FLAGS_t_ar);
            config::action_config.num_action_classes = actions_map.size();

            std::array<std::string, 7> outputBlobList;
            !config::action_config.new_network ? outputBlobList = {"mbox_loc1/out/conv/flat",
                                                                   "mbox_main_conf/out/conv/flat/softmax/flat",
                                                                   "mbox/priorbox",
                                                                   "out/anchor1",
                                                                   "out/anchor2",
                                                                   "out/anchor3",
                                                                   "out/anchor4" } 
                                               : outputBlobList = {"ActionNet/out_detection_loc",
                                                                   "ActionNet/out_detection_conf",
                                                                   "ActionNet/action_heads/out_head_1_anchor_1",
                                                                   "ActionNet/action_heads/out_head_2_anchor_1",
                                                                   "ActionNet/action_heads/out_head_2_anchor_2",
                                                                   "ActionNet/action_heads/out_head_2_anchor_3",
                                                                   "ActionNet/action_heads/out_head_2_anchor_4" };
            action_net = cv::gapi::ie::Params<custom::PersonDetActionRec>{
                ad_model_path,
                util::GetBinPath(ad_model_path),
                FLAGS_d_act,
            }.cfgOutputLayers(outputBlobList);
        }
        if (!fd_model_path.empty()) {
            // Load face detector
            config::face_config.confidence_threshold = static_cast<float>(FLAGS_t_fd);
            config::face_config.input_h = FLAGS_inh_fd;
            config::face_config.input_w = FLAGS_inw_fd;
            config::face_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
            config::face_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);

            det_net = cv::gapi::ie::Params<custom::FaceDetector>{
                fd_model_path,
                util::GetBinPath(fd_model_path),
                FLAGS_d_fd,
            };
        }

        if (!fd_model_path.empty() && !fr_model_path.empty() && !lm_model_path.empty()) {
            // Create face detection config for reid
            config::face_registration_det_config.confidence_threshold = static_cast<float>(FLAGS_t_reg_fd);
            config::face_registration_det_config.increase_scale_x = static_cast<float>(FLAGS_exp_r_fd);
            config::face_registration_det_config.increase_scale_y = static_cast<float>(FLAGS_exp_r_fd);

            landm_net = cv::gapi::ie::Params<custom::LandmarksDetector>{
                lm_model_path,
                util::GetBinPath(lm_model_path),
                FLAGS_d_lm,
            };

            reident_net = cv::gapi::ie::Params<custom::FaceReidentificator>{
                fr_model_path,
                util::GetBinPath(fr_model_path),
                FLAGS_d_reid,
            };

            /** TODO: Code from reid_gallary
              * needs put it back if possible
              */
            std::vector<int> idx_to_id;
            std::vector<GalleryObject> identities;
            if (!ids_list.empty()) {
                cv::GComputation gallery_pp([&]() {
                    cv::GMat in;
                    cv::GArray<cv::Rect> rects;

                    if (FLAGS_crop_gallery) {
                        cv::GMat detections =
                            cv::gapi::infer<custom::FaceDetector>(in);

                        cv::GArray<detection::DetectedObject> faces =
                            custom::FaceDetectorPostProc::on( detections
                                                            , in
                                                            , config::face_registration_det_config);
                        rects = custom::GetRectsFromDetections::on(faces);
                    }
                    else {
                        rects = custom::GetRectFromImage::on(in);
                    }

                    cv::GArray<cv::GMat> landmarks =
                        cv::gapi::infer<custom::LandmarksDetector>(rects, in);

                    cv::GMat align_face =
                        custom::AlignFacesForReidentification::on( landmarks
                                                                 , rects
                                                                 , in);

                    cv::GMat embeddings = cv::gapi::infer<custom::FaceReidentificator>(align_face);
                    return cv::GComputation(cv::GIn(in), cv::GOut(rects, embeddings));
                });

                auto gallery_kernels = cv::gapi::kernels< custom::OCVFaceDetectorPostProc
                                                        , custom::OCVAlignFacesForReidentification
                                                        , custom::OCVGetRectFromImage
                                                        , custom::OCVGetRectsFromDetections>();
                auto gallery_networks = cv::gapi::networks(det_net, landm_net, reident_net);

                cv::FileStorage fs(ids_list, cv::FileStorage::Mode::READ);
                cv::FileNode fn = fs.root();
                int id = 0;
                for (cv::FileNodeIterator fit = fn.begin(); fit != fn.end(); ++fit) {
                    cv::FileNode item = *fit;
                    std::string label = item.name();
                    std::vector<cv::Mat> embeddings;
            
                    // Please, note that the case when there are more than one image in gallery
                    // for a person might not work properly with the current implementation
                    // of the demo.
                    // Remove this assert by your own risk.
                    CV_Assert(item.size() == 1);
            
                    for (size_t i = 0; i < item.size(); i++) {
                        std::string path;
                        if (file_exists(item[i].string())) {
                            path = item[i].string();
                        }
                        else {
                            path = folder_name(ids_list) + separator() + item[i].string();
                        }
            
                        cv::Mat image = cv::imread(path);
                        CV_Assert(!image.empty());
                        cv::Mat emb;
                        std::vector<cv::Rect> rects;
                        gallery_pp.apply(cv::gin(image), cv::gout(rects, emb),
                                         cv::compile_args(gallery_kernels, gallery_networks));
                        emb = emb.reshape(1, { emb.size().width, 1 });

                        // NOTE: RegistrationStatus analog check
                        if (!rects.empty() &&
                            !(rects.size() > 1) &&
                            (rects[0].width > FLAGS_min_size_fr) &&
                            (rects[0].height > FLAGS_min_size_fr)) {
                            embeddings.push_back(emb);
                            idx_to_id.push_back(id);
                            identities.emplace_back(embeddings, label, id);
                            ++id;
                        }
                    }
                }
                slog::info << "Face reid gallery size: " << identities.size() << slog::endl;
            } else {
                slog::warn << "Face reid gallery is empty!" << slog::endl;
            }
            config::rec_config.reid_threshold = FLAGS_t_reid;
            config::rec_config.identities = identities;
            config::rec_config.idx_to_id = idx_to_id;
            config::rec_config.greedy_reid_matching = FLAGS_greedy_reid_matching;
            config::rec_config.reid_realizable = !fd_model_path.empty() &&
                                                 !fr_model_path.empty() &&
                                                 !lm_model_path.empty();
            config::rec_config.actions_type = actions_type;
        } else {
            slog::warn << "Face recognition models are disabled!" << slog::endl;
            if (actions_type == TEACHER) {
                slog::err << "Face recognition must be enabled to recognize teacher actions." << slog::endl;
                return 1;
            }
        }

        // Create tracker for reid
        TrackerParams tracker_reid_params;
        tracker_reid_params.min_track_duration = 1;
        tracker_reid_params.forget_delay = 150;
        tracker_reid_params.affinity_thr = 0.8f;
        tracker_reid_params.averaging_window_size_for_rects = 1;
        tracker_reid_params.averaging_window_size_for_labels = std::numeric_limits<int>::max();
        tracker_reid_params.bbox_heights_range = cv::Vec2f(10, 1080);
        tracker_reid_params.drop_forgotten_tracks = false;
        tracker_reid_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_reid_params.objects_type = "face";

        Tracker tracker_reid(tracker_reid_params);

        // Create Tracker for action recognition
        TrackerParams tracker_action_params;
        tracker_action_params.min_track_duration = 8;
        tracker_action_params.forget_delay = 150;
        tracker_action_params.affinity_thr = 0.9f;
        tracker_action_params.averaging_window_size_for_rects = 5;
        tracker_action_params.averaging_window_size_for_labels = FLAGS_ss_t > 0
                                                                 ? FLAGS_ss_t
                                                                 : actions_type == TOP_K ? 5 : 1;
        tracker_action_params.bbox_heights_range = cv::Vec2f(10, 2160);
        tracker_action_params.drop_forgotten_tracks = false;
        tracker_action_params.max_num_objects_in_track = std::numeric_limits<int>::max();
        tracker_action_params.objects_type = "action";

        Tracker tracker_action(tracker_action_params);

        cv::GComputation pp([&]() {
            cv::GMat in;
            cv::GMat frame = cv::gapi::copy(in);
            auto outs = GOut(frame);

            cv::GArray<cv::GMat> embeddings;
            cv::GArray<detection::DetectedObject> faces;
            if (actions_type != TOP_K) {
                if (!fd_model_path.empty()) {
                    cv::GMat detections =
                        cv::gapi::infer<custom::FaceDetector>(in);
                    faces = custom::FaceDetectorPostProc::on( detections
                                                            , in
                                                            , config::face_config);

                    if (!fr_model_path.empty() && !lm_model_path.empty()) {
                        cv::GArray<cv::Rect> rects =
                            custom::GetRectsFromDetections::on(faces);

                        cv::GArray<cv::GMat> landmarks =
                            cv::gapi::infer<custom::LandmarksDetector>(rects, in);

                        cv::GMat align_faces =
                            custom::AlignFacesForReidentification::on(landmarks, rects, in);

                        embeddings = cv::gapi::infer<custom::FaceReidentificator>(rects, align_faces);
                    } else {
                        embeddings = custom::GetEmptyMatGArray::on(in);
                    }
                } else {
                    faces = custom::GetEmptyFaces::on(in);
                    embeddings = custom::GetEmptyMatGArray::on(in);
                }
            }

            cv::GArray<TrackedObject> tracked_actions;
            cv::GArray<DetectedAction> persons_with_actions;

            if (!ad_model_path.empty()) {
                cv::GMat location;
                cv::GMat detect_confidences;
                cv::GMat priorboxes;
                cv::GMat action_con1;
                cv::GMat action_con2;
                cv::GMat action_con3;
                cv::GMat action_con4;

                std::tie( location
                        , detect_confidences
                        , priorboxes
                        , action_con1
                        , action_con2
                        , action_con3
                        , action_con4) = cv::gapi::infer<custom::PersonDetActionRec>(in);

                persons_with_actions =
                    custom::PersonDetActionRecPostProc::on( location
                                                          , detect_confidences
                                                          , priorboxes
                                                          , action_con1
                                                          , action_con2
                                                          , action_con3
                                                          , action_con4
                                                          , in
                                                          , config::action_config);
            } else {
                persons_with_actions = custom::GetEmptyActions::on(in);
            }

            if (actions_type != TOP_K) {
                cv::GArray<TrackedObject> tracked_faces;
                cv::GArray<std::string> face_labels;
                cv::GArray<std::string> face_id_to_label_map;
                cv::GArray<Track> face_tracks;

                std::tie( tracked_faces
                        , tracked_actions
                        , face_labels
                        , face_id_to_label_map
                        , face_tracks) = custom::GetRecognizeResult::on( embeddings
                                                                       , faces
                                                                       , persons_with_actions
                                                                       , in
                                                                       , config::rec_config);
                outs += GOut(tracked_actions, tracked_faces, face_labels, face_id_to_label_map, face_tracks);
            } else {
                tracked_actions = custom::GetActionTopHandsDetectionResult::on(persons_with_actions, in);
                outs += GOut(tracked_actions);
            }
            return cv::GComputation(cv::GIn(in), std::move(outs));
        });
        auto kernels = cv::gapi::kernels < custom::OCVFaceDetectorPostProc
                                         , custom::OCVGetRectsFromDetections
                                         , custom::OCVPersonDetActionRecPostProc
                                         , custom::OCVGetRecognizeResult
                                         , custom::OCVAlignFacesForReidentification
                                         , custom::OCVGetEmptyFaces
                                         , custom::OCVGetEmptyActions
                                         , custom::OCVGetEmptyMatGArray>();
        auto networks = cv::gapi::networks(det_net, landm_net, reident_net, action_net);

        cv::GStreamingCompiled cc;
        if (actions_type != TOP_K) {
            cc = pp.compileStreaming(cv::compile_args( kernels
                                                     , networks
                                                     , TrackerParamsPack{ tracker_reid_params
                                                                        , tracker_action_params }));
        } else {
            cc = pp.compileStreaming(cv::compile_args( cv::gapi::kernels < custom::OCVPersonDetActionRecPostProc
                                                                         , custom::OCVGetActionTopHandsDetectionResult
                                                                         , custom::OCVGetEmptyActions>()
                                                     , cv::gapi::networks(action_net)
                                                     , tracker_action_params));
        }

        cv::Mat frame;

        const char ESC_KEY = 27;
        // const char SPACE_KEY = 32;
        const cv::Scalar green_color(0, 255, 0);
        const cv::Scalar red_color(0, 0, 255);
        const cv::Scalar white_color(255, 255, 255);

        int teacher_track_id = -1;
        
        cv::VideoWriter vid_writer;
        Visualizer sc_visualizer(!FLAGS_no_show, vid_writer, num_top_persons);
        DetectionsLogger logger(std::cout, FLAGS_r, FLAGS_ad, FLAGS_al);

        std::cout << "To close the application, press 'CTRL+C' here";
        if (!FLAGS_no_show) {
            std::cout << " or switch to the output window and press ESC key";
        }
        std::cout << std::endl;

        cv::Size graphSize{static_cast<int>(frame.cols / 4), 60};
        Presenter presenter(FLAGS_u, frame.rows - graphSize.height - 10, graphSize);
        
        std::vector<std::string> face_id_to_label_map;
        std::vector<Track> face_tracks;
        std::vector<std::map<int, int>> face_obj_id_to_action_maps;
        std::map<int, int> top_k_obj_ids;

        // NOTE: 30 - hardcoded FPS (without cap.get(cv::CAP_PROP_FPS))
        const int smooth_window_size = static_cast<int>(30 * FLAGS_d_ad);
        const int smooth_min_length = static_cast<int>(30 * FLAGS_min_ad);

        util::Avg avg;

        video_path != "cam" ? cc.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(video_path)))
                            : cc.setSource(cv::gin(cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(0)));

        cc.start();
        avg.start();
        while (cc.running()) {
            logger.CreateNextFrameRecord(video_path, state::work_num_frames, frame.cols, frame.rows);
            char key = cv::waitKey(1);
            if (key == ESC_KEY) {
                break;
            }
            TrackedObjects tracked_actions;
            TrackedObjects tracked_faces;
            std::vector<std::string> face_labels;
            face_id_to_label_map.clear();
            face_tracks.clear();

            auto out_vector = cv::gout(frame, tracked_actions);
            if (actions_type != TOP_K) {
                out_vector += cv::gout( tracked_faces, face_labels
                                      , face_id_to_label_map, face_tracks);
            }

            if (!cc.pull(std::move(out_vector))) {
                if (cv::waitKey(1) >= 0) break;
                else continue;
            }

            if (!FLAGS_out_v.empty() && !vid_writer.isOpened()) {
                vid_writer = cv::VideoWriter( FLAGS_out_v
                                            , cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
                                            , 30
                                            , Visualizer::GetOutputSize(frame.size()));
            }

            presenter.handleKey(key);
            presenter.drawGraphs(frame);
            sc_visualizer.SetFrame(frame);

            if (actions_type == TOP_K) {
                if (static_cast<int>(top_k_obj_ids.size()) < FLAGS_a_top) {
                    for (const auto& action : tracked_actions) {
                        if (action.label == top_action_id && top_k_obj_ids.count(action.object_id) == 0) {
                            const int action_id_in_top = top_k_obj_ids.size();
                            top_k_obj_ids.emplace(action.object_id, action_id_in_top);
                
                            sc_visualizer.DrawCrop(action.rect, action_id_in_top, red_color);
                
                            if (static_cast<int>(top_k_obj_ids.size()) >= FLAGS_a_top) {
                                break;
                            }
                        }
                    }
                }
                
                avg.elapsed();
                ++state::work_num_frames;
                
                sc_visualizer.DrawFPS(static_cast<float>(avg.fps(state::work_num_frames)),
                                      red_color);
                
                for (const auto& action : tracked_actions) {
                    auto box_color = white_color;
                    std::string box_caption = "";
                
                    if (top_k_obj_ids.count(action.object_id) > 0) {
                        box_color = red_color;
                        box_caption = std::to_string(top_k_obj_ids[action.object_id] + 1);
                    }
                
                    sc_visualizer.DrawObject(action.rect, box_caption, white_color, box_color, true);
                }
            } else {
                std::map<int, int> frame_face_obj_id_to_action;
                for (size_t j = 0; j < tracked_faces.size(); j++) {
                    const auto& face = tracked_faces[j];
                    std::string face_label = face_labels[j];

                    std::string label_to_draw;
                    if (face.label != EmbeddingsGallery::unknown_id)
                        label_to_draw += face_label;

                    int person_ind = GetIndexOfTheNearestPerson(face, tracked_actions);
                    int action_ind = default_action_index;
                    if (person_ind >= 0) {
                        action_ind = tracked_actions[person_ind].label;
                    }

                    if (actions_type == STUDENT) {
                        if (action_ind != default_action_index) {
                            label_to_draw += "[" + GetActionTextLabel(action_ind, actions_map) + "]";
                        }
                        frame_face_obj_id_to_action[face.object_id] = action_ind;
                        sc_visualizer.DrawObject(face.rect, label_to_draw, red_color, white_color, true);
                        logger.AddFaceToFrame(face.rect, face_label, "");
                    }

                    if ((actions_type == TEACHER) && (person_ind >= 0)) {
                        if (face_label == teacher_id) {
                            teacher_track_id = tracked_actions[person_ind].object_id;
                        }
                        else if (teacher_track_id == tracked_actions[person_ind].object_id) {
                            teacher_track_id = -1;
                        }
                    }
                }
                if (actions_type == STUDENT) {
                    for (const auto& action : tracked_actions) {
                        const auto& action_label = GetActionTextLabel(action.label, actions_map);
                        const auto& action_color = GetActionTextColor(action.label);
                        const auto& text_label = fd_model_path.empty() ? action_label : "";
                        sc_visualizer.DrawObject(action.rect, text_label, action_color, white_color, true);
                        logger.AddPersonToFrame(action.rect, action_label, "");
                        logger.AddDetectionToFrame(action, state::work_num_frames);
                    }
                    face_obj_id_to_action_maps.push_back(frame_face_obj_id_to_action);
                }
                else if (teacher_track_id >= 0) {
                    auto res_find = std::find_if(tracked_actions.begin(), tracked_actions.end(),
                        [teacher_track_id](const TrackedObject& o) { return o.object_id == teacher_track_id; });
                    if (res_find != tracked_actions.end()) {
                        const auto& tracker_action = *res_find;
                        const auto& action_label = GetActionTextLabel(tracker_action.label, actions_map);
                        sc_visualizer.DrawObject(tracker_action.rect, action_label, red_color, white_color, true);
                        logger.AddPersonToFrame(tracker_action.rect, action_label, teacher_id);
                    }
                }
                avg.elapsed();
                sc_visualizer.DrawFPS(static_cast<float>(avg.fps(state::work_num_frames)),
                                      red_color);
                ++state::work_num_frames;
            }

            ++state::total_num_frames;

            sc_visualizer.Show();

            if (FLAGS_last_frame >= 0 && state::work_num_frames > static_cast<size_t>(FLAGS_last_frame)) {
                break;
            }
            logger.FinalizeFrameRecord();
        }
        sc_visualizer.Finalize();

        slog::info << slog::endl;
        if (state::work_num_frames > 0) {
            slog::info << "Mean FPS: " << static_cast<float>(state::work_num_frames) /
                                          static_cast<float>(avg.elapsed().ss) << slog::endl;
        }
        slog::info << "Frames processed: " << state::total_num_frames << slog::endl;

        if (actions_type == STUDENT) {        
            // correct labels for track
            std::vector<Track> new_face_tracks = UpdateTrackLabelsToBestAndFilterOutUnknowns(face_tracks);
            std::map<int, int> face_track_id_to_label = GetMapFaceTrackIdToLabel(new_face_tracks);
        
            if (!face_id_to_label_map.empty()) {
                std::map<int, FrameEventsTrack> face_obj_id_to_actions_track;
                ConvertActionMapsToFrameEventTracks(face_obj_id_to_action_maps, default_action_index,
                                                    &face_obj_id_to_actions_track);
        
                const int start_frame = 0;
                const int end_frame = face_obj_id_to_action_maps.size();
                std::map<int, RangeEventsTrack> face_obj_id_to_events;
                SmoothTracks(face_obj_id_to_actions_track, start_frame, end_frame,
                             smooth_window_size, smooth_min_length, default_action_index,
                             &face_obj_id_to_events);
        
                slog::info << "Final ID->events mapping" << slog::endl;
                logger.DumpTracks(face_obj_id_to_events,
                                  actions_map, face_track_id_to_label,
                                  face_id_to_label_map);
        
                std::vector<std::map<int, int>> face_obj_id_to_smoothed_action_maps;
                ConvertRangeEventsTracksToActionMaps(end_frame, face_obj_id_to_events,
                                                     &face_obj_id_to_smoothed_action_maps);
        
                slog::info << "Final per-frame ID->action mapping" << slog::endl;
                logger.DumpDetections(video_path, frame.size(), state::work_num_frames,
                                      new_face_tracks,
                                      face_track_id_to_label,
                                      actions_map, face_id_to_label_map,
                                      face_obj_id_to_smoothed_action_maps);
            }
        }

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