// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <opencv2/gapi/gkernel.hpp>

namespace kp {
    cv::gapi::GKernelPackage gallery_kernels();
    cv::gapi::GKernelPackage video_process_kernels();
    cv::gapi::GKernelPackage top_k_kernels();
} //namespace kp
