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
