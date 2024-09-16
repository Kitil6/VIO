#include <camera.hpp>
#include <frame.hpp>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud.h>
/*
    * 0: cam
    * 1: dataset
    * 2: topic
*/
#define methed  2
#define dataset 0


cv::Mat img_rec;
bool flag_rec = false;
void rec_msg(const sensor_msgs::ImageConstPtr &msg){
    img_rec = cv_bridge::toCvCopy(msg, "bgr8")->image;
    flag_rec = true;
}

void get_img(cv::Mat &img){
    while (flag_rec == false){
        ros::spinOnce();
        // printf("wating for imamge.....\n");
        if(ros::ok() == 0)
            break;
    }
    flag_rec = false;
    cv::cvtColor(img_rec, img, cv::COLOR_BGR2GRAY);
}

std::string base_addr_0 = "/home/kitil/dataset/dataset_0/";
std::string base_addr_1 = "/home/kitil/dataset/dataset_1/";
bool get_picture(cv::Mat &img, std::ifstream &pics){
    std::string line;
    if(std::getline(pics, line)){
        std::stringstream ss_rgb(line);
        getline(ss_rgb, line, ' ');
        getline(ss_rgb, line, ' ');
        cv::Mat color;
        if(dataset == 0)
            color = cv::imread(base_addr_0 + line);
        else if(dataset == 1){
            getline(ss_rgb, line, ' ');
            getline(ss_rgb, line, ' ');
            color = cv::imread(base_addr_1 + line);
        }
        cv::cvtColor(color, img, cv::COLOR_BGR2GRAY);
        return false;
    }
    return true;
}

int main(int argc, char **argv){
    ros::init(argc, argv, "VO");
    ros::NodeHandle nh;
    tf::TransformBroadcaster broadcaster;
    tf::Transform transform;
    ros::Subscriber image_sub = nh.subscribe("/iris_0/stereo_camera/left/image_raw", 1, rec_msg);
    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud>("cloud", 1);

    cv::VideoCapture cap(0);
    std::string addr;
    if(dataset == 0)
        addr = "/home/kitil/dataset/dataset_0/rgb.txt";
    else if(dataset == 1)
        addr = "/home/kitil/dataset/dataset_1/rgb.txt";
    std::ifstream fin_rgb(addr);

    cv::Mat img1, img2;
    bool state = 1;
    int cnt = 0;
    Camera camera;
    Frame frame;
    unsigned len = 0;
    if(methed == 1){
        std::string line;
        for (int i = 0; i < 3; i++){
            std::getline(fin_rgb, line);
        }
        get_picture(img1, fin_rgb);
    }
    else if(methed == 0){
        for(int i=0; i<30; i++)
            camera.get_img(img1, cap);
    }
    else if(methed == 2){
        get_img(img1);
    }

    std::ofstream cloud_("/home/kitil/VO_exercise/src/VO/src/cloud.txt", std::ios::trunc);    
    if (cloud_.is_open())
        cloud_.close();
    while (ros::ok()){
        // 初始化
        if(state){
            if(methed == 1)
                get_picture(img2, fin_rgb);
            else if(methed == 0)
                camera.get_img(img2, cap);
            else if(methed == 2)
                get_img(img2);
            state = frame.VO_initial(img1, img2, cnt);
            cnt++;
            img1 = img2.clone();
            if (state == 0){
                std::cout << "############ Initial success ############" << std::endl;
                continue;
            }
            
        }
        else{
            /*====================================================================*/
            sensor_msgs::PointCloud cloud_p;
            cloud_p.header.stamp = ros::Time::now();
            cloud_p.header.frame_id = "world";
            cloud_p.points.resize(frame.pts_curre_frame.xyz.size());

            cloud_p.channels.resize(1);
            cloud_p.channels[0].name = "rgb";
            cloud_p.channels[0].values.resize(frame.pts_curre_frame.xyz.size());
            
            for(unsigned int i = 0; i < frame.pts_curre_frame.xyz.size(); ++i){
                cloud_p.points[i].x = frame.pts_curre_frame.xyz[i].z;
                cloud_p.points[i].y = -frame.pts_curre_frame.xyz[i].x;
                cloud_p.points[i].z = frame.pts_curre_frame.xyz[i].y;
                cloud_p.channels[0].values[i] = 255;
            }
            cloud_pub.publish(cloud_p);
            /*====================================================================*/

            if(methed == 1)
                get_picture(img2, fin_rgb);
            else if(methed == 0)
                camera.get_img(img2, cap);
            else if(methed == 2)
                get_img(img2);
            frame.VO_motion_tracking(img1, img2);
            img1 = img2.clone();
            // ######### draw point ######### //
            cv::Mat img = img2.clone();
            for (auto i : frame.pts_curre_frame.uv){
                cv::circle(img, i, 10, (255, 255, 255));
            }
            cv::imshow("Points", img);
            cv::waitKey(50);
            // ######### draw point ######### //
            std::ofstream cloud("/home/kitil/VO_exercise/src/VO/src/cloud.txt", std::ios::app);
            if(frame.cloud.xyz.size() > len){
                for(; len<frame.cloud.xyz.size(); len++)
                    cloud << frame.cloud.xyz[len].x << ','
                          << frame.cloud.xyz[len].y << ','
                          << frame.cloud.xyz[len].z << '\n'; 
            }
            cloud.close();
        }
        Eigen::Matrix3f R;
        R << frame.T_cur(0, 0), frame.T_cur(0, 1), frame.T_cur(0, 2), 
             frame.T_cur(1, 0), frame.T_cur(1, 1), frame.T_cur(1, 2), 
             frame.T_cur(2, 0), frame.T_cur(2, 1), frame.T_cur(2, 2);

        Eigen::Quaternionf q(R);

        if(std::isnan(q.x()) || std::isnan(q.y()) || std::isnan(q.z()) || std::isnan(q.w()));
        else if(abs(frame.T_cur(1, 3))+abs(frame.T_cur(2, 3))+abs(frame.T_cur(0, 3)) < 10000){
            transform.setOrigin(tf::Vector3(-frame.T_cur(2, 3)/1, frame.T_cur(0, 3)/1, frame.T_cur(1, 3)/1));
            transform.setRotation(tf::Quaternion(q.x(), q.y(), q.z(), q.w()));
            broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "base_link"));
        }
            
        ros::spinOnce();
    }

    return 0;
}