#include <camera.hpp>

cv::Point2f Camera::pixel_2_norm(const cv::Point2f &pt){
    cv::Point2f p;
    p.x = (pt.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
    p.y = (pt.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
    return p;
}


void Camera::get_img(cv::Mat & img, cv::VideoCapture &cap){
    cv::Mat color;
    cap >> color;
    cv::cvtColor(color, img, cv::COLOR_BGR2GRAY);
}
