#include "frame.hpp"
#include "camera.hpp"
#include <ceres/ceres.h>
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>

Eigen::Matrix4d Frame::Rt_2_T(const cv::Mat &R, const cv::Mat &t){
    Eigen::Matrix4d T;
    T <<    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
            R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
            R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
            0, 0, 0, 1;
    return T;
}
Eigen::Vector4d Frame::point3f_2_vector4d(const cv::Point3f &pt){
    Eigen::Vector4d p(pt.x, pt.y, pt.z, 1.0);
    return p;
}

Eigen::Vector3d Frame::point2f_2_vector3d(const cv::Point2f &pt){
    Eigen::Vector3d p(pt.x, pt.y, 1.0);
    return p;
}

cv::Point2f Frame::uv_2_xy(const cv::Point2f &uv){
    cv::Point2f xy;
    xy.x = (uv.x - K.at<double>(0, 2)) / K.at<double>(0, 0);
    xy.y = (uv.y - K.at<double>(1, 2)) / K.at<double>(1, 1);
    return xy;
}

void Frame::point_filter_after_tracking(cv::Mat &img, std::vector<cv::Point2f> &uv1, std::vector<cv::Point2f> &uv2, int len){
    std::vector<cv::Point2f> uv1_good, uv2_good;
    int col_up = img.cols - len;
    int row_up = img.rows - len;
    for(int i=0; i<uv1.size(); i++){
        if((uv1[i].x<col_up && uv1[i].x>len) && (uv1[i].y<row_up && uv1[i].y>len))
            if((uv2[i].x<col_up && uv2[i].x>len) && (uv2[i].y<row_up && uv2[i].y>len)){
                uv1_good.push_back(uv1[i]);
                uv2_good.push_back(uv2[i]);
            }
    }
    uv1 = uv1_good;
    uv2 = uv2_good;
}

void Frame::point_filter_after_tracking(cv::Mat &img, std::vector<cv::Point2f> &uv2, std::vector<cv::Point3f> &xyz2, int len){
    std::vector<cv::Point2f> uv2_good;
    std::vector<cv::Point3f> xyz2_good;
    int col_up = img.cols - len;
    int row_up = img.rows - len;
    for(int i=0; i<uv2.size(); i++){
            if((uv2[i].x<col_up && uv2[i].x>len) && (uv2[i].y<row_up && uv2[i].y>len)){
                uv2_good.push_back(uv2[i]);
                xyz2_good.push_back(xyz2[i]);
            }
    }
    uv2 = uv2_good;
    xyz2 = xyz2_good;
}


// 检测三角测量的点 向前投影和向后投影的误差
std::vector<cv::Point3f> Frame::check_points(const cv::Mat &Proj1, const cv::Mat &Proj2, std::vector<cv::Point2f> &xy1, std::vector<cv::Point2f> &xy2, std::vector<cv::Point2f> &uv1, std::vector<cv::Point2f> &uv2, cv::Mat &xyz, double threshold){
    Eigen::Matrix<double, 4, 4> T1, T2;
    T1 << Proj1.at<float>(0, 0), Proj1.at<float>(0, 1), Proj1.at<float>(0, 2), Proj1.at<float>(0, 3),
          Proj1.at<float>(1, 0), Proj1.at<float>(1, 1), Proj1.at<float>(1, 2), Proj1.at<float>(1, 3),
          Proj1.at<float>(2, 0), Proj1.at<float>(2, 1), Proj1.at<float>(2, 2), Proj1.at<float>(2, 3),
          0, 0, 0, 1;
    T2 << Proj2.at<float>(0, 0), Proj2.at<float>(0, 1), Proj2.at<float>(0, 2), Proj2.at<float>(0, 3),
          Proj2.at<float>(1, 0), Proj2.at<float>(1, 1), Proj2.at<float>(1, 2), Proj2.at<float>(1, 3),
          Proj2.at<float>(2, 0), Proj2.at<float>(2, 1), Proj2.at<float>(2, 2), Proj2.at<float>(2, 3),
          0, 0, 0, 1;
    std::vector<Eigen::Vector4d> xyz_h;
    for (int i = 0; i < xyz.cols; i++){
        xyz.col(i) /= xyz.col(i).at<float>(3, 0);
        xyz_h.push_back(Eigen::Vector4d(xyz.col(i).at<float>(0, 0), xyz.col(i).at<float>(1, 0), xyz.col(i).at<float>(2, 0), 1.0));
    }
    std::vector<double> err1, err2;
    for(int i=0; i<xy1.size();i++){
        double err;
        Eigen::Vector4d TP1 = T1*xyz_h[i];
        if(TP1(2) < 0 || TP1(2) > 100){
            err2.push_back(999.0);
            err1.push_back(999.0);
            continue;
        }
        TP1 /= TP1(2);
        err = abs(xy1[i].x-TP1(0)) + abs(xy1[i].y-TP1(1));
        err1.push_back(err);
        // std::cout << "img1:";
        // std::cout << err << '\n';
        Eigen::Vector4d TP2 = T2*xyz_h[i];
        if(TP2(2) < 0 || TP2(2) > 100){
            err2.push_back(999.0);
            continue;
        }
        TP2 /= TP2(2);
        err = abs(xy2[i].x-TP2(0)) + abs(xy2[i].y-TP2(1));
        err2.push_back(err);
        // std::cout << "img2:";
        // std::cout << err << '\n';
    }
    std::vector<double> err;
    for(int i=0; i<err1.size(); i++){
        double err_all = err1[i] + err2[i];
        err.push_back(err_all);
        // std::cout << err_all << '\n';
    }
    std::vector<double> err_;
    double err_sum, err_mean, err_midl;
    for(int i=0; i<err1.size(); i++){
        double e = err[i];
        err_.push_back(e);
        err_sum += e;
    }
    err_mean = err_sum/err_.size();
    std::sort(err_.begin(), err_.end());
    err_midl = err_[err_.size()/2];
    for(auto i : err_)
        std::cout << i << "\n";

    std::vector<cv::Point2f> uv1_, uv2_;
    std::vector<cv::Point3f> pts;
    for(int i=0; i<xy1.size();i++){
        if(err[i] < threshold){
            uv1_.push_back(uv1[i]);
            uv2_.push_back(uv2[i]);
            pts.push_back(cv::Point3f(xyz_h[i](0), xyz_h[i](1), xyz_h[i](2)));
        }
    }
    uv1 = uv1_;
    uv2 = uv2_;
    return pts;
}

double Frame::cal_parallax(Eigen::Matrix4d T_pre, Eigen::Matrix4d T_cur){
    return sqrt((T_pre(0, 3)-T_cur(0, 3))*(T_pre(0, 3)-T_cur(0, 3)) + 
                (T_pre(1, 3)-T_cur(1, 3))*(T_pre(1, 3)-T_cur(1, 3)) + 
                (T_pre(2, 3)-T_cur(2, 3))*(T_pre(2, 3)-T_cur(2, 3)) );
}

/*
    1. 提取特征点
    2. 光流跟踪
    3. 存储跟踪成功的特征点
    4. 计算平均每对特征点之间的”像素距离“, 若小于15，则不进行初始化
    5. 根据每对特征点（八对，采取随机采样一致性策略），求解本质矩阵 E 
    6. 根据本质矩阵E = t^R, 进行SVD分解，求解出两组解，对应四种可能性，只有一个结果的深度大于0
    7. 根据 R 和 t 进行三角测量，求解出特征点的空间坐标，（世界坐标系下）
*/
bool Frame::VO_initial(cv::Mat & img1, cv::Mat & img2, int &cnt){
    std::vector<cv::Point2f> pts_cur, pts_pre;
    std::vector<unsigned char> status;
    std::vector<float> err;
    std::vector<cv::Point2f> uv_pre, uv_cur;
    if(cnt == 0){
        cv::goodFeaturesToTrack(img1, pts_pre, 80, 0.02, 30);
        std::cout << "初始化中，初始特征点： " << pts_pre.size() << '\n';
        cv::calcOpticalFlowPyrLK(img1, img2, pts_pre, pts_cur, status, err);
        for(int i=0; i<(int)status.size(); i++){
            if(status[i]){
                init_uv.push_back(pts_pre[i]);
                uv_cur.push_back(pts_cur[i]);
            }
        }
        std::cout << "初始化中，当前剩余： " << uv_cur.size() << '\n';
    }
    else{
        cv::calcOpticalFlowPyrLK(img1, img2, pts_last_frame.uv, pts_cur, status, err);
        std::list<cv::Point2f>::iterator it = init_uv.begin();
        for(int i=0; i<(int)status.size(); i++){
            if(status[i]){
                uv_cur.push_back(pts_cur[i]);
                it++;
            }
            else{
                it = init_uv.erase(it);
            }
        }
    }
    pts_last_frame.uv = uv_cur;
    std::cout << "初始化中，当前剩余： " << uv_cur.size() << '\n';
    // ######### draw point ######### //
    cv::Mat img = img2.clone();
    for (auto i : uv_cur)
    {
        cv::circle(img, i, 10, (255, 255, 255));
    }
    cv::imshow("init", img);
    cv::waitKey(50);
    // ######### draw point ######### //

    if(cnt>20 || (uv_cur.size()<40)){
        for(std::list<cv::Point2f>::iterator it=init_uv.begin(); it!=init_uv.end(); ++it){
            uv_pre.push_back(*it);
        }
        point_filter_after_tracking(img1, uv_pre, uv_cur, 20);
        cv::Point2d principal_point(K.at<double>(0, 2), K.at<double>(1, 2));
        double focal_length = K.at<double>(0, 0);
        // 计算E = t^R
        cv::Mat essential_matrix = cv::findEssentialMat(uv_pre, uv_cur, focal_length, principal_point);
        // 奇异直分解，求解t^ 与 R， 且 t 已经归一化了
        cv::Mat R, t;
        try{
            cv::recoverPose(essential_matrix, uv_pre, uv_cur, K, R, t);
        }
        catch (cv::Exception){
            return true;
        }
        
        std::cout << "R = " << R << std::endl;
        std::cout << "t = " << t << std::endl;

        // 估计特征点深度
        cv::Mat T1 = (cv::Mat_<float>(3, 4) <<  1, 0, 0, 0,
                                                0, 1, 0, 0,
                                                0, 0, 1, 0);
        cv::Mat T2 = (cv::Mat_<float>(3, 4) <<  R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                                                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                                                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));

        // 像素坐标系 ——> 归一化平面坐标系
        std::vector<cv::Point2f> xy_pre, xy_cur;  // 因为当前跟踪到的特征点在下次跟踪还要用，不能转换为归一化平面坐标
        xy_pre.reserve(uv_pre.size());
        xy_cur.reserve(uv_pre.size());
        for (int i = 0; i < (int)uv_pre.size(); i++){
            xy_pre.push_back(uv_2_xy(uv_pre[i]));
            xy_cur.push_back(uv_2_xy(uv_cur[i]));
        }
        cv::Mat pts_4d;
        cv::triangulatePoints(T1, T2, xy_pre, xy_cur, pts_4d);

        pts_last_frame.uv.clear();
        pts_last_frame.xyz.clear();
        pts_curre_frame.uv.clear();
        pts_curre_frame.xyz.clear();
        std::cout << "处理之前：" << xy_pre.size() << ',' << xy_cur.size() << '\n';
        pts_last_frame.xyz = pts_curre_frame.xyz = check_points(T1, T2, xy_pre, xy_cur, uv_pre, uv_cur, pts_4d, 0.08);
        pts_curre_frame.uv = uv_cur;
        pts_last_frame.uv = uv_pre;
        std::cout << "处理之后：" << uv_pre.size() << ',' << uv_cur.size() << '\n';
        // ####### 可以向前投影，或者向后投影，来检测特征点的正确性 ####### // 
        cloud = pts_curre_frame;
        pts_last_frame = pts_curre_frame;
        T_pre << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
                0, 0, 0, 1;
        return false;
    }
    return true;
}


void Frame::feature_tracking(cv::Mat & img1, cv::Mat & img2){
    std::vector<cv::Point2f> pts_cur;
    std::vector<unsigned char> status;
    std::vector<float> err;
    try{
        cv::calcOpticalFlowPyrLK(img1, img2, pts_last_frame.uv, pts_cur, status, err);
    }catch(cv::Exception){
        std::cout << "特征点不足！！！！!!!!!!!!!!!!!!" << std::endl;
    }
    pts_curre_frame.uv.clear();
    pts_curre_frame.xyz.clear();
    for(int i=0; i<(int)status.size(); i++){
        if(status[i]){
            pts_curre_frame.uv.push_back(pts_cur[i]);
            pts_curre_frame.xyz.push_back(pts_last_frame.xyz[i]);
        }
    }
    std::vector<cv::Point2f> pts_pre, pts_cur_;
    std::vector<cv::Point3f> xyz;
    std::vector<unsigned char> status1;
    std::vector<float> err1;
    cv::calcOpticalFlowPyrLK(img2, img1, pts_curre_frame.uv, pts_pre, status1, err1);
    for(int i=0; i<(int)status1.size(); i++){
        if(status1[i]){
            pts_cur_.push_back(pts_curre_frame.uv[i]);
            xyz.push_back(pts_curre_frame.xyz[i]);
        }
    }
    pts_curre_frame.uv = pts_cur_;
    pts_curre_frame.xyz = xyz;
    
    point_filter_after_tracking(img1, pts_curre_frame.uv, pts_curre_frame.xyz, 20);

    // std::cout << "上一帧的特征点数量： " << pts1.size() << '\n';
    // std::cout << "当前跟踪到的特征点数量： " << pts2.size() << '\n';
}

void Frame::feature_new(cv::Mat & img1, cv::Mat & img2){
    std::vector<unsigned char> status;
    std::vector<float> err;
    std::vector<cv::Point2f> pts_pre, pts_cur;
    
    if((pts_curre_frame.uv.size() < MAX_CNT) && (uv_new.second.first.size() == 0)){
        uv_new.second.second.clear();
        // std::cout << "当前新增的特征点数量： " << 100-pts2.size() << '\n';
        cv::Mat mask = cv::Mat::zeros(img1.size(), CV_8UC1);
        Camera::set_mask(pts_curre_frame, mask, MASK_R);
        cv::goodFeaturesToTrack(img1, pts_pre, MAX_CNT-pts_curre_frame.uv.size()+10, 0.02, minDistance, mask);
        cv::calcOpticalFlowPyrLK(img1, img2, pts_pre, pts_cur, status, err);
        std::vector<cv::Point2f> uv_pre, uv_cur;
        uv_pre.reserve(100);
        uv_cur.reserve(100);
        for(int i=0; i<(int)status.size(); i++){
            if(status[i]){
                uv_pre.push_back(pts_pre[i]);
                uv_cur.push_back(pts_cur[i]);
            }
        }
        point_filter_after_tracking(img1, uv_pre, uv_cur, 20);
        uv_new.first = T_cur;
        for(int i=0; i<uv_cur.size(); i++){
            uv_new.second.first.push_back(uv_pre[i]);
            uv_new.second.second.push_back(uv_cur[i]);
        }
        // std::cout << "======================= 等待新增：" << uv_cur.size() << '\n';
    }
    // printf("======================= 当前parallax: %lf \n", cal_parallax(uv_new.first, T_cur) );
    if(uv_new.second.first.size() > 0){
        std::vector<unsigned char> status;
        std::vector<float> err;
        std::vector<cv::Point2f> pts_pre, pts_cur, uv_cur;
        std::list<cv::Point2f>::iterator it_cur = uv_new.second.second.begin();
        std::list<cv::Point2f>::iterator it_init = uv_new.second.first.begin();
        for (; it_cur != uv_new.second.second.end(); ++it_cur)
            pts_pre.push_back(*it_cur);
        cv::calcOpticalFlowPyrLK(img1, img2, pts_pre, pts_cur, status, err);
        
        it_cur = uv_new.second.second.begin();
        for(int i=0; i< status.size(); i++){
            if(status[i]){
                ++it_cur;
                ++it_init;
                uv_cur.push_back(pts_cur[i]);
            }
            else{
                it_cur = uv_new.second.second.erase(it_cur);
                it_init = uv_new.second.first.erase(it_init);
            }
        }

        if(cal_parallax(uv_new.first, T_cur) > parallax || pts_curre_frame.uv.size() < 10){
            std::vector<cv::Point2f> uv_pre;
            for(std::list<cv::Point2f>::iterator it=uv_new.second.first.begin(); it!=uv_new.second.first.end(); ++it){
                uv_pre.push_back(*it);
            }
            
            if(uv_cur.size() > 0){
                Eigen::Matrix4d delta_T = T_cur * T_pre.inverse();
                T_pre = uv_new.first;
                cv::Mat delta_Proj2 = (cv::Mat_<float>(3, 4) <<  delta_T(0, 0), delta_T(0, 1), delta_T(0, 2), delta_T(0, 3),
                                                        delta_T(1, 0), delta_T(1, 1), delta_T(1, 2), delta_T(1, 3),
                                                        delta_T(2, 0), delta_T(2, 1), delta_T(2, 2), delta_T(2, 3) );

                cv::Mat Proj1 = (cv::Mat_<float>(3, 4) <<  T_pre(0, 0), T_pre(0, 1), T_pre(0, 2), T_pre(0, 3),
                                                           T_pre(1, 0), T_pre(1, 1), T_pre(1, 2), T_pre(1, 3),
                                                           T_pre(2, 0), T_pre(2, 1), T_pre(2, 2), T_pre(2, 3) );
                cv::Mat Proj2 = (cv::Mat_<float>(3, 4) <<  T_cur(0, 0), T_cur(0, 1), T_cur(0, 2), T_cur(0, 3),
                                                           T_cur(1, 0), T_cur(1, 1), T_cur(1, 2), T_cur(1, 3),
                                                           T_cur(2, 0), T_cur(2, 1), T_cur(2, 2), T_cur(2, 3) );
                cv::Mat pts_4d;
                // 像素坐标系 ——> 归一化平面坐标系
                std::vector<cv::Point2f> xy_pre, xy_cur;  // 因为当前跟踪到的特征点在下次跟踪还要用，不能转换为归一化平面坐标
                xy_pre.reserve(uv_pre.size());
                xy_cur.reserve(uv_pre.size());
                for (int i = 0; i < (int)uv_pre.size(); i++){
                    xy_pre.push_back(uv_2_xy(uv_pre[i]));
                    xy_cur.push_back(uv_2_xy(uv_cur[i]));
                }
                std::cout << "处理之前：" << uv_pre.size() << ',' << uv_cur.size() << '\n';
                cv::triangulatePoints(Proj1, Proj2, xy_pre, xy_cur, pts_4d);
                std::vector<cv::Point3f> xyz;
                xyz = check_points(Proj1, Proj2, xy_pre, xy_cur, uv_pre, uv_cur, pts_4d, 0.08);
                for(int i=0; i<xyz.size(); i++){
                    pts_curre_frame.xyz.push_back(xyz[i]);
                    pts_curre_frame.uv.push_back(uv_cur[i]);
                    cloud.uv.push_back(uv_cur[i]);
                    cloud.xyz.push_back(xyz[i]);
                }
                std::cout << "处理之后：" << uv_pre.size() << ',' << uv_cur.size() << '\n';
                std::cout << "当前总共特征点：" << pts_curre_frame.xyz.size() << '\n';
                uv_new.second.first.clear();
                uv_new.second.second.clear();
            }
        }
    }
}


// Eigen::Matrix<double, 9, 1> delta_x(const cv::Mat &K, const std::vector<Eigen::Vector3d> &TP, const cv::Mat &R, const std::vector<Eigen::Vector2d> &uv);
Eigen::Matrix<double, 6, 1> delta_x(const cv::Mat &K, const std::vector<Eigen::Vector3d> &TP, const std::vector<Eigen::Vector2d> &uv);
template<typename T>
T delta(const cv::Mat &K, const std::vector<Eigen::Vector3d> &TP, const cv::Mat &R, const std::vector<Eigen::Vector2d> &uv);

void Frame::VO_motion_tracking(cv::Mat & img1, cv::Mat & img2){
    feature_tracking(img1, img2);       
    // double dist = Camera::calc_dist(pts_last_frame.uv, pts_curre_frame.uv);
    cv::Mat r, t, R;
    cv::solvePnP(pts_curre_frame.xyz, pts_curre_frame.uv, K, cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(r, R);
    T_cur = Rt_2_T(R, t);
    // std::cout << t << '\n';
    // std::cout << r << '\n';
    /*=====================================================================================================================*/
    int cnt = 50;
    while(cnt){
        cnt--;
        clock_t start = clock();
        std::vector<Eigen::Vector3d> TP;
        std::vector<Eigen::Vector2d> uv;
        pts_curre_frame.xyz[pts_curre_frame.xyz.size()-1];
        for(int i=0; i<pts_curre_frame.xyz.size(); i++){
            Eigen::Vector4d TP_4 = T_cur*point3f_2_vector4d(pts_curre_frame.xyz[i]);
            TP.push_back(Eigen::Vector3d(TP_4(0), TP_4(1), TP_4(2)));
            uv.push_back(Eigen::Vector2d(pts_curre_frame.uv[i].x, pts_curre_frame.uv[i].y)); 
        }

        // Eigen::Matrix<double, 6, 1> x = delta_x(K, TP, uv);
        Eigen::MatrixXd x = delta<Eigen::MatrixXd>(K, TP, R, uv);
        if(std::isnan(x(0, 0))){
            T_cur = T_pre;
            // break;
            while(1)
                std::cout << x.transpose() << '\n';
        }
        // 截止当前，算出来的是p的增量，而不是t的增量，需要先将t->p，再p+delta_p, 再转换回 t
        t.at<double>(0, 0) += x(0); t.at<double>(1, 0) += x(1); t.at<double>(2, 0) += x(2);
        r.at<double>(0, 0) += x(3); r.at<double>(1, 0) += x(4); r.at<double>(2, 0) += x(5);
        cv::Rodrigues(r, R);
        // t = Jp, 由罗德里格斯公式给出J
        Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        double theta = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2));
        Eigen::Vector3d a(x(0)/theta, x(1)/theta, x(2)/theta);
        Eigen::Matrix3d a_s;
        a_s << 0,    -a(2),  a(1),
               a(2),  0   , -a(0),
               -a(1), a(0),  0;
        Eigen::Matrix3d J = sin(theta)/theta*I - (1-sin(theta)/theta)*a*a.transpose() + (1-cos(theta))/theta*a_s;
        Eigen::Vector3d t_(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
        Eigen::Vector3d p;
        Eigen::Matrix3d J_i = J.inverse();
        if(std::isnan(J_i(0, 0))){
            Eigen::FullPivLU<Eigen::Matrix3d> LU(J);
            J_i = LU.inverse();
        }
        p = J_i*t_;
        p(0) += 0.5*x(0); p(1) += 0.5*x(1); p(2) += 0.5*x(2);
        t_ = J*p;

        r.at<double>(0, 0) += 0.5*x(3); r.at<double>(1, 0) += 0.5*x(4); r.at<double>(2, 0) += 0.5*x(5);
        t.at<double>(0, 0) = t_(0); t.at<double>(1, 0) = t_(1); t.at<double>(2, 0) = t_(2);
        T_cur = Rt_2_T(R, t);
        for(int i=6; i< x.size(); i+=3){
            // std::cout << (i-6)/3 << '\n';
            pts_curre_frame.xyz[(i-6)/3].x += 0.5 * x(i);
            pts_curre_frame.xyz[(i-6)/3].y += 0.5 * x(i+1);
            pts_curre_frame.xyz[(i-6)/3].z += 0.5 * x(i+2);
        }
        clock_t end = clock();
        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        // std::cout << "==============迭代一次花费: " << elapsed*1000 << " ms" << std::endl;
        double del = x.norm(); 
        // std::cout << "当前状态更新量：" << del << '\n';
        if(del < 1e-4) break;
    }
    // std::cout << t << '\n';
    // std::cout << r << '\n';
    /*=====================================================================================================================*/

    double delta_t = cal_parallax(T_pre, T_cur);
    // if(t_n > 30)
    feature_new(img1, img2);
    pts_last_frame = pts_curre_frame;
    T_pre = T_cur;
    // std::cout << T_pre << '\n';
}


Eigen::Matrix<double, 6, 2>  calc_J_delta_T(const cv::Mat &K, const Eigen::Vector3d &TP){
    Eigen::Matrix<double, 6, 2> J;
    J << K.at<double>(0, 0)/TP(2),                                         0, 
         0,                                                                K.at<double>(1, 1)/TP(2),      
         -K.at<double>(0, 0)*TP(0)/TP(2)/TP(2),                           -K.at<double>(1, 1)*TP(1)/TP(2)/TP(2),
         -K.at<double>(0, 0)*TP(0)*TP(1)/TP(2)/TP(2),                     -K.at<double>(1, 1)-K.at<double>(1, 1)*TP(1)*TP(1)/TP(2)/TP(2),
         K.at<double>(0, 0) + K.at<double>(0, 0)*TP(0)*TP(0)/TP(2)/TP(2),  K.at<double>(1, 1)*TP(0)*TP(1)/TP(2)/TP(2),
         -K.at<double>(0, 0)*TP(1)/TP(2),                                  K.at<double>(1, 1)*TP(0)/TP(2);
    // J << K.at<double>(0, 0)/TP(2), 0,                        -K.at<double>(0, 0)*TP(0)/TP(2)/TP(2), -K.at<double>(0, 0)*TP(0)*TP(1)/TP(2)/TP(2),                    K.at<double>(0, 0) + K.at<double>(0, 0)*TP(0)*TP(0)/TP(2)/TP(2), -K.at<double>(0, 0)*TP(1)/TP(2),
    //      0                      , K.at<double>(1, 1)/TP(2),  -K.at<double>(1, 1)*TP(1)/TP(2)/TP(2), -K.at<double>(1, 1)-K.at<double>(1, 1)*TP(1)*TP(1)/TP(2)/TP(2), K.at<double>(1, 1)*TP(0)*TP(1)/TP(2)/TP(2)                     , K.at<double>(1, 1)*TP(0)/TP(2);
    return -J;
}

Eigen::Matrix<double, 2, 3>  calc_J_delta_P(const cv::Mat &K, const Eigen::Vector3d &TP, const cv::Mat &R){
    Eigen::Matrix<double, 2, 3> J;
    Eigen::Matrix3d R_;
    J << K.at<double>(0, 0) / TP(2),   0,                           -K.at<double>(0, 0) * TP(0) / TP(2) / TP(2),
        0,                             K.at<double>(1, 1) / TP(2),   -K.at<double>(1, 1) * TP(1) / TP(2) / TP(2);

    R_ << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    return -J*R_;
}

Eigen::Matrix<double, 6, 1> calc_g_delta_T(const Eigen::Matrix<double, 6, 2> &J, const Eigen::Vector2d &e){
    Eigen::Matrix<double, 6, 1> g = -J*e;
    return g;
}
Eigen::Matrix<double, 3, 1> calc_g_delta_P(const Eigen::Matrix<double, 3, 2> &J, const Eigen::Vector2d &e){
    Eigen::Matrix<double, 3, 1> g = -J*e;
    return g;
}
Eigen::Matrix<double, 9, 1> calc_g(const Eigen::Matrix<double, 2, 9> &J, const Eigen::Vector2d &e){
    Eigen::Matrix<double, 9, 1> g = -J.transpose()*e;
    return g;
}



Eigen::Matrix<double, 6, 1> delta_x(const cv::Mat &K, const std::vector<Eigen::Vector3d> &TP, const std::vector<Eigen::Vector2d> &uv){
    std::vector<Eigen::Matrix<double, 6, 2>> J_delta_T;
    std::vector<Eigen::Matrix<double, 6, 1>> G_delta_T;

    for(int i=0; i<uv.size(); i++){
        Eigen::Matrix<double, 6, 2> j_delta_T = calc_J_delta_T(K, TP[i]);
        Eigen::Vector2d e(uv[i](0) - K.at<double>(0, 0)*TP[i](0)/TP[i](2) - K.at<double>(0, 2), 
                          uv[i](1) - K.at<double>(1, 1)*TP[i](1)/TP[i](2) - K.at<double>(1, 2)
                         );

        J_delta_T.push_back(j_delta_T);
        G_delta_T.push_back(calc_g_delta_T(j_delta_T, e));

    }
    Eigen::MatrixXd JJT_M(6*J_delta_T.size(), 6);
    Eigen::VectorXd g_M  (6*G_delta_T.size());

    for(int i=0; i<J_delta_T.size(); i++){
        JJT_M.block<6, 6>(i*6, 0) = J_delta_T[i]*J_delta_T[i].transpose();
        g_M.block<6, 1>(i*6, 0) = G_delta_T[i];
    }

    // Eigen::HouseholderQR<Eigen::MatrixXd> qr(JJT_M);
    // qr.compute(JJT_M);
    // Eigen::Matrix<double, 6, 1> x = qr.solve(g_M);
    Eigen::Matrix<double, 6, 1> x = JJT_M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g_M);
    return x;
}



template<typename T>
T delta(const cv::Mat &K, const std::vector<Eigen::Vector3d> &TP, const cv::Mat &R, const std::vector<Eigen::Vector2d> &uv){
    std::vector<Eigen::Matrix<double, 2, 6>> J_delta_T;
    std::vector<Eigen::Matrix<double, 2, 3>> J_delta_P;
    std::vector<Eigen::Matrix<double, 9, 1>> G;
    std::vector<Eigen::Matrix<double, 2, 1>> E;
    for(int i=0; i<uv.size(); i++){
        Eigen::Matrix<double, 6, 2> j_delta_T = calc_J_delta_T(K, TP[i]);
        Eigen::Matrix<double, 2, 3> j_delta_P = calc_J_delta_P(K, TP[i], R);
        Eigen::Vector2d e(uv[i](0) - K.at<double>(0, 0)*TP[i](0)/TP[i](2) - K.at<double>(0, 2), 
                          uv[i](1) - K.at<double>(1, 1)*TP[i](1)/TP[i](2) - K.at<double>(1, 2)
                         );
        E.push_back(e);
        J_delta_T.push_back(j_delta_T.transpose());
        J_delta_P.push_back(j_delta_P); 
    }
    Eigen::MatrixXd J_M(2*J_delta_T.size(), 6 + 3*J_delta_P.size());
    Eigen::VectorXd E_M(2*uv.size());
    J_M.setZero();
    E_M.setZero();
    for(int i=0; i<J_delta_T.size(); i++){
        Eigen::MatrixXd J_i(2, 6 + 3*J_delta_P.size());
        J_i.setZero();
        J_i.block<2, 6>(0, 0) = J_delta_T[i];
        J_i.block<2, 3>(0, 6 + 3*i) = J_delta_P[i];
        Eigen::Vector2d e(uv[i](0) - K.at<double>(0, 0)*TP[i](0)/TP[i](2) - K.at<double>(0, 2), 
                          uv[i](1) - K.at<double>(1, 1)*TP[i](1)/TP[i](2) - K.at<double>(1, 2)
                         );
        J_M.block(i*2, 0, 2, 6 + 3*J_delta_P.size()) = J_i;
        E_M.block<2, 1>(i*2, 0) = E[i];
    }
    Eigen::VectorXd g_M = -J_M.transpose() * E_M;
    Eigen::MatrixXd JJT_M = J_M.transpose()*J_M;
    // Eigen::SparseMatrix<double> sparse_JJT_M = JJT_M.sparseView();
    // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> qr;
    // qr.compute(sparse_JJT_M);
    // T x = qr.solve(g_M);
    // std::cout << "JTJ SIZE: " << JJT_M.rows() << "x" << JJT_M.cols() << '\n';
    // std::cout << "g SIZE: " << g_M.rows() << "x" << g_M.cols() << '\n';
    // std::cout << "Determinant of JJT_M: " << JJT_M.determinant() << std::endl;
    // T x = JJT_M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g_M);
    T x = JJT_M.completeOrthogonalDecomposition().solve(g_M);
    // std::cout << J_M << '\n';
    // std::cout << "aaaa" <<'\n';
    return x;
}