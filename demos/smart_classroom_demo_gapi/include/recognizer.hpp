// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "face_reid.hpp"

struct FaceRecognizerConfig {
    double reid_threshold;
    std::vector<GalleryObject> identities;
    std::vector<int> idx_to_id;
    bool greedy_reid_matching;
};

class FaceRecognizer {
public:
    FaceRecognizer(FaceRecognizerConfig config)
        : face_gallery(config.reid_threshold,
                       config.identities,
                       config.idx_to_id,
                       config.greedy_reid_matching) {}

    bool LabelExists(const std::string &label) const {
        return face_gallery.LabelExists(label);
    }

    std::string GetLabelByID(int id) const {
        return face_gallery.GetLabelByID(id);
    }

    std::vector<std::string> GetIDToLabelMap() const {
        return face_gallery.GetIDToLabelMap();
    }

    std::vector<int> Recognize(std::vector<cv::Mat>& embeddings,
        const detection::DetectedObjects& faces) {
        if (embeddings.empty()) {
            return std::vector<int>(faces.size(), EmbeddingsGallery::unknown_id);
        }
        for (auto & emb : embeddings) {
            emb = emb.reshape(1, { 256, 1 });
        }
        return face_gallery.GetIDsByEmbeddings(embeddings);
    }

private:
    EmbeddingsGallery face_gallery;
};

struct FaceRecognizerKernelInput {
    std::shared_ptr<FaceRecognizer> ptr;
    double actions_type = 0;
};
