// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/videoio.hpp>
#include <opencv2/gapi/garg.hpp>

namespace cv {
namespace gapi {
namespace wip {
class CustomCapSource : public IStreamSource
{
public:
    explicit CustomCapSource(const cv::VideoCapture& cap) : cap(cap) { prep(); }

protected:
    cv::VideoCapture cap;
    cv::Mat first;
    bool first_pulled = false;
    cv::Mat clear_frame;
    void prep() {
        GAPI_Assert(first.empty());
        cv::Mat tmp;
        if (!cap.read(tmp)) {
            GAPI_Assert(false && "Couldn't grab the frame");
        }
        first = tmp.clone();
    }

    virtual bool pull(cv::gapi::wip::Data &data) override {
        if (!first_pulled) {
            GAPI_Assert(!first.empty());
            first_pulled = true;
            data = first;
            return true;
        }
        if (!cap.isOpened()) return false;
        cv::Mat frame;
        if (!cap.read(frame)) {
            return false;
        }
        data = frame.clone();
        return true;
    }

    virtual GMetaArg descr_of() const override {
        GAPI_Assert(!first.empty());
        return cv::GMetaArg{ cv::descr_of(first) };
    }
};
} // namespace wip
} // namespace gapi
} // namespace cv
