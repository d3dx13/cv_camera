// Copyright [2015] Takashi Ogura<t.ogura@gmail.com>

#include "cv_camera/capture.h"
#include <sstream>
#include <string>

namespace cv_camera {

    namespace enc = sensor_msgs::image_encodings;

    Capture::Capture(ros::NodeHandle &node, const std::string &topic_name,
                     int32_t buffer_size, const std::string &frame_id,
                     const std::string &camera_name)
            : node_(node),
              it_(node_),
              topic_name_(topic_name),
              buffer_size_(buffer_size),
              frame_id_(frame_id),
              info_manager_(node_, camera_name),
              capture_delay_(ros::Duration(node_.param("capture_delay", 0.0))) {
        // Skammi Extension
        // Get the flip parameters
        node_.param("flip_image", flip_image_, flip_image_);
        node_.param("image_flip_code", image_flip_code_, image_flip_code_);

        node_.param("undistorted_on", undistorted_on_, undistorted_on_);
        node_.param("undistorted_fov_scale", undistorted_fov_scale_, undistorted_fov_scale_);
        node_.param("undistorted_resolution_scale", undistorted_resolution_scale_, undistorted_resolution_scale_);
    }

    void Capture::loadCameraInfo() {
        std::string url;
        if (node_.getParam("camera_info_url", url)) {
            if (info_manager_.validateURL(url)) {
                info_manager_.loadCameraInfo(url);
            }
        }

        rescale_camera_info_ = node_.param<bool>("rescale_camera_info", false);

        for (int i = 0;; ++i) {
            int code = 0;
            double value = 0.0;
            std::stringstream stream;
            stream << "property_" << i << "_code";
            const std::string param_for_code = stream.str();
            stream.str("");
            stream << "property_" << i << "_value";
            const std::string param_for_value = stream.str();
            if (!node_.getParam(param_for_code, code) || !node_.getParam(param_for_value, value)) {
                break;
            }
            if (!cap_.set(code, value)) {
                ROS_ERROR_STREAM("Setting with code " << code << " and value " << value << " failed"
                                                      << std::endl);
            }
        }
    }

    void Capture::rescaleCameraInfo(int width, int height) {
        double width_coeff = static_cast<double>(width) / info_original_.width;
        double height_coeff = static_cast<double>(height) / info_original_.height;
        info_original_.width = width;
        info_original_.height = height;

        // See http://docs.ros.org/api/sensor_msgs/html/msg/CameraInfo.html for clarification
        info_original_.K[0] *= width_coeff;
        info_original_.K[2] *= width_coeff;
        info_original_.K[4] *= height_coeff;
        info_original_.K[5] *= height_coeff;

        info_original_.P[0] *= width_coeff;
        info_original_.P[2] *= width_coeff;
        info_original_.P[5] *= height_coeff;
        info_original_.P[6] *= height_coeff;
    }

    void Capture::undistort(std::string distortion_model) {
        info_.K[0] *= undistorted_resolution_scale_ / undistorted_fov_scale_;
        info_.K[4] *= undistorted_resolution_scale_ / undistorted_fov_scale_;

        info_.P[0] *= undistorted_resolution_scale_ / undistorted_fov_scale_;
        info_.P[5] *= undistorted_resolution_scale_ / undistorted_fov_scale_;

        info_.K[2] *= undistorted_resolution_scale_;
        info_.K[5] *= undistorted_resolution_scale_;

        info_.P[2] *= undistorted_resolution_scale_;
        info_.P[6] *= undistorted_resolution_scale_;

        info_.width = (int) round(image_.cols * undistorted_resolution_scale_);
        info_.height = (int) round(image_.rows * undistorted_resolution_scale_);

        double D_array[4];
        std::copy(info_original_.D.begin(), info_original_.D.begin() + 4, D_array);

        cv::Mat D(1, 4, CV_64F, D_array);
        cv::Mat K(3, 3, CV_64F, info_original_.K.c_array());
        cv::Mat K_new(3, 3, CV_64F, info_.K.c_array());

        if (undistorted_map_recalculate) {
            cv::Size dim_new;
            dim_new.height = info_.height;
            dim_new.width = info_.width;
            if (distortion_model.compare("fisheye") == 0) {
                cv::fisheye::initUndistortRectifyMap(K, D, cv::Mat(), K_new, dim_new, CV_16SC2, undistorted_map1,
                                                     undistorted_map2);
            } else {
                cv::initUndistortRectifyMap(K, D, cv::Mat(), K_new, dim_new, CV_16SC2, undistorted_map1,
                                            undistorted_map2);
            }

            resize(image_, bridge_.image, dim_new, cv::INTER_LINEAR);

            undistorted_map_recalculate = false;
        }

        cv::remap(image_, bridge_.image, undistorted_map1, undistorted_map2, cv::INTER_LINEAR,
                  cv::BORDER_CONSTANT, cv::Scalar());

        double D_array_zero[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
        info_.D = cv::Mat(1, 5, CV_64F, D_array_zero);
        info_.distortion_model = "none";
    }

    void Capture::open(int32_t device_id) {
        cap_.open(device_id);
        if (!cap_.isOpened()) {
            std::stringstream stream;
            stream << "device_id" << device_id << " cannot be opened";
            throw DeviceError(stream.str());
        }
        pub_ = it_.advertiseCamera(topic_name_, buffer_size_);

        loadCameraInfo();
    }

    void Capture::open(const std::string &device_path) {
        cap_.open(device_path, cv::CAP_V4L);
        if (!cap_.isOpened()) {
            throw DeviceError("device_path " + device_path + " cannot be opened");
        }
        pub_ = it_.advertiseCamera(topic_name_, buffer_size_);

        loadCameraInfo();
    }

    void Capture::open() {
        open(0);
    }

    void Capture::openFile(const std::string &file_path) {
        cap_.open(file_path);
        if (!cap_.isOpened()) {
            std::stringstream stream;
            stream << "file " << file_path << " cannot be opened";
            throw DeviceError(stream.str());
        }
        pub_ = it_.advertiseCamera(topic_name_, buffer_size_);

        std::string url;
        if (node_.getParam("camera_info_url", url)) {
            if (info_manager_.validateURL(url)) {
                info_manager_.loadCameraInfo(url);
            }
        }
    }

    bool Capture::capture() {
        if (cap_.read(image_)) {
            ros::Time stamp = ros::Time::now() - capture_delay_;
            bridge_.encoding = image_.channels() == 3 ? enc::BGR8 : enc::MONO8;
            bridge_.header.stamp = stamp;
            bridge_.header.frame_id = frame_id_;

            // Skammi extension
            // Flip the image?
            if (flip_image_) {
                cv::flip(image_, image_, image_flip_code_);
            }

            info_original_ = info_manager_.getCameraInfo();

            if (info_original_.height == 0 && info_original_.width == 0) {
                info_original_.height = bridge_.image.rows;
                info_original_.width = bridge_.image.cols;
            } else if (info_original_.height != bridge_.image.rows ||
                       info_original_.width != bridge_.image.cols) {
                if (rescale_camera_info_) {
                    int old_width = info_original_.width;
                    int old_height = info_original_.height;
                    rescaleCameraInfo(image_.cols, image_.rows);
                    ROS_INFO_ONCE("Camera calibration automatically rescaled from %dx%d to %dx%d",
                                  old_width, old_height, image_.cols, image_.rows);
                } else {
                    ROS_WARN_ONCE("Calibration resolution %dx%d does not match camera resolution %dx%d. "
                                  "Use rescale_camera_info param for rescaling",
                                  info_original_.width, info_original_.height, image_.cols, image_.rows);
                }
            }
            info_original_.header.stamp = stamp;
            info_original_.header.frame_id = frame_id_;

            info_ = info_original_;

            std::stringstream distortion_models_ss;
            distortion_models_ss << info_original_.distortion_model;
            std::string distortion_model_str = distortion_models_ss.str();
            boost::algorithm::to_lower(distortion_model_str);

            if (undistorted_on_ and info_manager_.isCalibrated()) {
                undistort(distortion_model_str);
            } else {
                bridge_.image = image_;
            }

            return true;
        }
        return false;
    }

    void Capture::publish() {
        pub_.publish(*getImageMsgPtr(), info_);
    }

    bool Capture::setPropertyFromParam(int property_id, const std::string &param_name) {
        if (cap_.isOpened()) {
            double value = 0.0;
            if (node_.getParam(param_name, value)) {
                ROS_INFO("setting property %s = %lf", param_name.c_str(), value);
                return cap_.set(property_id, value);
            }
        }
        return true;
    }

} // namespace cv_camera
