#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <future>
#include <functional>
#include <chrono>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>
#include "line_detector.h"

using namespace std;
using namespace cv;

class LaneAssistant
{
    // insert your custom functions and algorithms here
public:
    LaneAssistant()
    {
    }

    bool processData( tronis::CircularMultiQueuedSocket& socket )
    {
        string control_cmd = std::to_string( steer_output_norm_ ) + ";" +
                             std::to_string( throttle_output_norm_ ) + ";" +
                             std::to_string( isCardAhead ) + ";" + formatDouble( distance_ );

        socket.send( tronis::SocketData( control_cmd ) );

        return true;
    }

    void processImage( cv::Mat img )
    {
        original_image_ = img;
        image_ = img;
        is_stop_lane_present_ = true;
        isRedLightPresent( image_ );

        cv::Mat region_of_interest = regionDetection( image_ );

        cv::Mat color_filtered_img = colorDetection( region_of_interest );

        cv::Mat edge_detected_img = edgeDetection( color_filtered_img );

        cv::Mat bird_eye_view_img = birdEyeView( edge_detected_img, true );

        //laneDetection_withBEV( bird_eye_view_img, image_ );
        imshow( "bird eye view", bird_eye_view_img );
        imshow( "color detection", color_filtered_img );
        imshow( "color detection", image_ );
    }

protected:
    // Image
    std::string image_name_;
    cv::Mat image_, original_image_, BEV_image_;

    // Ego Vehicle
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;

    // Lane Detection
    vector<Point> left_lane_, right_lane_, stop_lane_;
    vector<double> left_lane_slope_history_, right_lane_slope_history_;
    vector<double> left_lane_bias_history, right_lane_bias_history;
    Mat left_lane_coeff_, right_lane_coeff_, stop_lane_coeff;
    int processed_frames = 0;
    Lane right_lane, left_lane;

    // Stop Sign and Stop Lane Detection
    bool is_red_sign_present_ = false;
    bool is_stop_lane_present_ = false;
    bool shouldBrake_ = false;    // need to brake or not
    bool isStopHandled_ = false;  // check if the car has stopped for duration
    std::chrono::steady_clock::time_point startTime_;
    std::chrono::steady_clock::time_point currentTime_;
    // hyperparameter of steering PID controller
    double steer_output_norm_ = 0.0;
    double steer_P_ = 2.0;
    double steer_I_ = 0.0001;
    double steer_D_ = 10;

    // Initialize error, derivative of error, integration of error
    double steer_error_old_ = 0;
    double steer_error_I_ = 0;

    // Initialize throttle PID controller and parameters
    double throttle_output_norm_;
    double max_velocity = 50;  // max velocity
    double throttle_P = 0.04;
    double throttle_I = 125 / 1e6;
    double throttle_D = -0.02;
    double throttle_error_old = 0;
    double throttle_error_I = 0;

    // Initialize distance PID controller and parameters
    bool isCardAhead;
    double distance_ = std::numeric_limits<double>::infinity();
    double min_safe_distance_ = 30;  // min safe car-car distance
    double max_detected_distance_ = 60.0;
    double distance_P = 1.5;
    double distance_I = 15 / 1e6;
    double distance_D = 0.0003;
    double distance_error_old = 0;
    double distance_error_I = 0;

    std::map<std::string, int> rightLaneVehicle;
    std::map<std::string, int> leftLaneVehicle;

    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        is_stop_lane_present_ = true;
        isRedLightPresent( image_ );

        cv::Mat region_of_interest = regionDetection( image_ );

        cv::Mat color_filtered_img = colorDetection( region_of_interest );

        cv::Mat edge_detected_img = edgeDetection( color_filtered_img );

        laneDetection( edge_detected_img, image_ );

        steeringControl( image_ );
        
        cv::Mat bird_eye_view_img = birdEyeView( color_filtered_img, true );
        imshow( "lane detected", image_ );
        imshow( "bird eye view", bird_eye_view_img );
        /* cv::putText( bird_eye_view_img, "Steering:" + to_string( steer_output_norm_ ),
                      Point( 300, 45 ), FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 255, 0 ), 1 );*/
        throttleControl();
    }

    /* Aufgabe2: Lane Detection*/

    // Create region of interest
    cv::Mat regionDetection( Mat img )
    {
        if( img.empty() )
        {
            cout << "No Image Input!" << endl;
        }
        Mat result;
        int img_height = img.size().height;
        int img_width = img.size().width;
        // mask image with polygon
        Mat mask_crop( img_height, img_width, CV_8UC1, Scalar( 0 ) );

        // mask points
        vector<Point> pts;
        vector<vector<Point>> v_pts;
        pts.push_back( Point( 0, img_height / 2 + 45 ) );          // top left
        pts.push_back( Point( img_width, img_height / 2 + 45 ) );  // top right
        pts.push_back( Point( img_width, img_height - 70 ) );      // bottom left
        // pts.push_back( Point( img_width / 2, img_height - 80 ) );  // bottom center
        pts.push_back( Point( 0, img_height - 70 ) );  // bottom right

        v_pts.push_back( pts );

        /// add cone to mask
        fillPoly( mask_crop, v_pts, ( 255, 255, 255 ) );

        if( img.type() == CV_8UC3 )
        {
            Mat mask_three_channel;
            cv::cvtColor( mask_crop, mask_three_channel,
                          COLOR_GRAY2BGR );  // transform mask into three channels
            cv::bitwise_and( img, mask_three_channel, result );
            // cout << " Region is detected." << endl;
        }
        else
        {
            Mat mask_three_channel( mask_crop.size(),
                                    CV_8UC4 );  // why here should be four channels

            vector<Mat> channels( 4 );
            channels[0] = mask_crop;
            channels[1] = mask_crop;
            channels[2] = mask_crop;
            channels[3] = mask_crop;
            merge( channels, mask_three_channel );

            cv::bitwise_and( img, mask_three_channel, result );
            // cout << " Region is detected." << endl;
        }

        return result;
    }

    // Filter white line with HSV color space
    cv::Mat colorDetection( cv::Mat img )
    {
        /*According to HSV color space, this function will pick white and yellow region from
         * original region*/
        Mat mask_yellow, mask_white, mask_combined, img_hsv;
        // Transform image from RBG to HSV color space
        cvtColor( img, img_hsv, COLOR_BGR2HSV );

        // white region hsv threshold
        cv::Scalar lower_white( 106, 0, 120 );
        cv::Scalar upper_white( 173, 90, 200 );
        cv::inRange( img_hsv, lower_white, upper_white, mask_white );

        // yellow region hsv threshold
        cv::Scalar lower_yellow( 0, 150, 75 );
        cv::Scalar upper_yellow( 160, 255, 255 );
        cv::inRange( img_hsv, lower_yellow, upper_yellow, mask_yellow );

        int yellow_pixels = cv::countNonZero( mask_yellow );
        // cout << "Current yellow pixels are " << yellow_pixels << endl;

        if( yellow_pixels > 100 )
        {
            // imshow( "yellow mask", mask_yellow );
            return mask_yellow;
        }
        else
        {
            return mask_white;
        }
        imshow( "color detection", mask_white );

        // Combine two regions
        // cv::bitwise_or( mask_yellow, mask_white, mask_combined );
    }

    // Use Canny() to detect edges
    cv::Mat edgeDetection( cv::Mat image )
    {
        cv::Mat grad_img, blurred_img;

        /* First method: sobel operator
        cv::Mat src, src_gray, grad;
        int ddepth = CV_16S;

        // Reduce noise
        cv::GaussianBlur( image, src, cv::Size( 3, 3 ), 0, 0, cv::BORDER_DEFAULT );

        // Convert the image to grayscale
        cv::cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

        // Sobel Operatpr
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        cv::Sobel( src_gray, grad_x, ddepth, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );
        cv::Sobel( src_gray, grad_y, ddepth, 1, 0, 1, 1, 0, cv::BORDER_DEFAULT );

        // Convert image to a CV_8U
        cv::convertScaleAbs( grad_x, abs_grad_x );
        cv::convertScaleAbs( grad_y, abs_grad_y );

        // Gradient
        cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );*/

        // Second method: Canny Edge Detector
        cv::GaussianBlur( image, blurred_img, cv::Size( 3, 3 ), 0, 0, cv::BORDER_DEFAULT );

        cv::Canny( image, grad_img, 100, 200 );

        // cout << " Edge is detected." << endl;
        return grad_img;
    }

    cv::Mat birdEyeView( Mat img, bool mode )
    {
        /*Create bird eye view*/

        int image_height = img.size().height;
        int image_width = img.size().width;

        vector<Point2f> src( 4 ), dst( 4 );
        // src[0] = Point2f( 0, image_height / 2 + 45 );            // top left
        // src[1] = Point2f( 0, image_height - 70 );                // bottom left
        // src[2] = Point2f( image_width, image_height - 70 );      // bottom right
        // src[3] = Point2f( image_width, image_height / 2 + 45 );  // top right

        // dst[0] = Point2f( 0, 0 );                                // top left
        // dst[1] = Point2f( image_width / 2 - 75, image_height );  // bottom left
        // dst[2] = Point2f( image_width / 2 + 75, image_height );  // bottom right
        // dst[3] = Point2f( image_width, 0 );                      // top right

        src[0] = Point2f( image_width / 2 - 150, image_height / 2 + 45 );  // top left
        src[1] = Point2f( 0, image_height - 100 );                         // bottom left
        src[2] = Point2f( image_width, image_height - 100 );               // bottom right
        src[3] = Point2f( image_width / 2 + 150, image_height / 2 + 45 );  // top right

        dst[0] = Point2f( image_width / 2 - 250, 0 );             // top left
        dst[1] = Point2f( image_width / 2 - 200, image_height );  // bottom left
        dst[2] = Point2f( image_width / 2 + 200, image_height );  // bottom right
        dst[3] = Point2f( image_width / 2 + 250, 0 );             // top right
        Mat M = getPerspectiveTransform( src, dst );
        Mat Minv = getPerspectiveTransform( dst, src );
        Mat BEV_img;

        if( mode == true )
        {
            warpPerspective( img, BEV_img, M, Size( image_width, image_height ) );
        }
        else
        {
            warpPerspective( img, BEV_img, Minv, Size( image_width, image_height ) );
        }

        return BEV_img;
    }

    // Lane Detection
    void laneDetection( cv::Mat img, cv::Mat output_img )
    {
        std::vector<Vec4i> raw_lines;
        cv::Mat img_color;
        cv::HoughLinesP( img, raw_lines, 6, CV_PI / 180, 100, 40,
                         10 );  // Probabilistic Line Transform 1 and CV_PI:resolution , 50 :
                                // threshold of number of votes, 50 : minLineLength, 10: maxLineGap

        if( raw_lines.empty() )
        {
            cout << "no line is detected!" << endl;
        }

        cvtColor( img, img_color, COLOR_GRAY2BGR );
        // painting lines in original image
        for( size_t i = 0; i < raw_lines.size(); i++ )
        {
            Vec4i l = raw_lines[i];

            line( img_color, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3,
                  LINE_AA );
        }

        imshow( "raw_lines", img_color );

        vector<Point> left_lines, right_lines, stop_lines;  // the points of left and right lanes
        cv::cvtColor( img, img_color, COLOR_GRAY2BGR );

        // devide all detected lines into 3 groups accroding to slope
        for( const Vec4i& l : raw_lines )
        {
            double slope = static_cast<double>( l[3] - l[1] ) / ( l[2] - l[0] );
            double bias = l[3] - slope * l[2];

            if( slope > 0.3 && slope < 0.75
                //&& bias > 60 && bias < 170
            )
            {
                right_lines.emplace_back( l[0], l[1] );
                right_lines.emplace_back( l[2], l[3] );
            }
            else if( slope < -0.2 && slope > -0.75
                     //&& bias > 350 && bias < 450
            )
            {
                left_lines.emplace_back( l[0], l[1] );
                left_lines.emplace_back( l[2], l[3] );
            }
            else if( -0.1 < slope && slope < 0.1 )
            {
                stop_lines.emplace_back( l[0], l[1] );
                stop_lines.emplace_back( l[2], l[3] );
            }
        }

        // merge and draw right lane
        if( !right_lines.empty() )
        {
            right_lane_ = mergeLines( right_lines );
            laneDetectionRobust( right_lane_, "r" );
            //saveToCSV( right_lane_, true );
            line( output_img, right_lane_.at( 0 ), right_lane_.at( 1 ), Scalar( 0, 255, 0 ), 5,
                  LINE_AA );
        }
        /*else
        {
            line( output_img, right_lane_.at( 0 ), right_lane_.at( 1 ), Scalar( 255, 255, 0 ), 5,
                  LINE_AA );
            cout << "Right lane is not detected, previous detection will be used." << endl;
        }*/
        // merge and draw left lane
        if( !left_lines.empty() )
        {
            left_lane_ = mergeLines( left_lines );
            laneDetectionRobust( left_lane_, "l" );
            //saveToCSV( left_lane_, false );

            line( output_img, left_lane_.at( 0 ), left_lane_.at( 1 ), Scalar( 255, 0, 0 ), 5,
                  LINE_AA );
        }

        /*else
        {
            line( output_img, left_lane_.at( 0 ), left_lane_.at( 1 ), Scalar( 0, 0, 255 ), 5,
                  LINE_AA );
            cout << "Left lane is not detected, previous detection will be used." << endl;
        }*/

        // Draw region of detection
        std::vector<Point> poly_pts;
        poly_pts.push_back( left_lane_.at( 0 ) );
        poly_pts.push_back( right_lane_.at( 0 ) );
        poly_pts.push_back( right_lane_.at( 1 ) );
        poly_pts.push_back( cv::Point( output_img.cols, output_img.rows ) );
        poly_pts.push_back( cv::Point( 0, output_img.rows ) );
        poly_pts.push_back( left_lane_.at( 1 ) );

        std::vector<vector<Point>> poly_line{poly_pts};

        cv::Mat img_poly = Mat( output_img.size(), CV_8UC4, Scalar( 0, 0, 0, 0.5 ) );

        cv::fillPoly( img_poly, poly_line, cv::Scalar( 0, 255, 100 ) );
        addWeighted( output_img, 1.0, img_poly, 0.2, 0.0, output_img );

        // Merge stop lane
        if( !stop_lines.empty() )
        {
            stop_lane_ = mergeLines( stop_lines );
        }
    }

    /*
     * merge all line which point to same direction to one line
     *
     * @param points the endpoints on the line
     * @return the endpoints of the merged line
     */
    vector<Point> mergeLines( const vector<Point>& points )
    {
        Vec4f line_parameters;
        int img_height = original_image_.rows;
        /* fitLine(input vector, output line, distance type, distance parameter, radial
         parameter, angle parameter) output (vx, vy, x, y)*/
        fitLine( points, line_parameters, DIST_L2, 0, 0.01, 0.01 );

        Point point1, point2;
        point1.x = line_parameters[2];
        point1.y = line_parameters[3];
        double k = line_parameters[1] / line_parameters[0];

        //
        int startY = img_height / 2 + 40;
        point1.x = point1.x + ( startY - point1.y ) / k;
        point1.y = startY;

        //
        if( k > 0 )  // right lane points
        {
            point2.x = original_image_.cols;
        }
        else
        {
            point2.x = 0;
        }

        point2.y = k * ( point2.x - point1.x ) + point1.y;

        vector<Point> lane_points;
        lane_points.push_back( point1 );
        lane_points.push_back( point2 );

        //
        // line( image, point1, point2, color, 2, LINE_AA );

        // Save all the lane points
        /*vector<Point> points_lane;
        for( int i = img_height / 2 + 40; i < img_height; i++ )
        {
            points_lane.push_back(
                Point( point1.x + ( i - point1.y ) / line_parameters[1] * line_parameters[0], i ) );
        }*/

        return lane_points;
    }

    void laneDetectionRobust( vector<Point> lane, string lane_name )
    {
        vector<double>* lane_slope_history_ptr = nullptr;
        vector<double>* lane_bias_history_ptr = nullptr;

        if( lane_name == "r" )
        {
            lane_slope_history_ptr = &right_lane_slope_history_;
            lane_bias_history_ptr = &right_lane_bias_history;
        }
        else
        {
            lane_slope_history_ptr = &left_lane_slope_history_;
            lane_bias_history_ptr = &left_lane_bias_history;
        }

        // Define slope and bias
        double slope = static_cast<double>( lane.at( 1 ).y - lane.at( 0 ).y ) /
                       ( lane.at( 1 ).x - lane.at( 0 ).x );

        double bias = static_cast<double>( lane.at( 1 ).y - slope * lane.at( 1 ).x );

        // Fill slope and bias history vector
        if( lane_slope_history_ptr->size() < 6 )
        {
            lane_slope_history_ptr->push_back( slope );
            lane_bias_history_ptr->push_back( bias );
        }
        // When have enough slope and bias history
        else
        {
            lane_slope_history_ptr->erase( lane_slope_history_ptr->begin() );
            lane_slope_history_ptr->push_back( slope );

            lane_bias_history_ptr->erase( lane_bias_history_ptr->begin() );
            lane_bias_history_ptr->push_back( bias );

            double slope_sum = std::accumulate( lane_slope_history_ptr->begin(),
                                                lane_slope_history_ptr->end(), 0.0 );
            double slope_mean = slope_sum / lane_slope_history_ptr->size();

            double bias_sum = std::accumulate( lane_bias_history_ptr->begin(),
                                               lane_bias_history_ptr->end(), 0.0 );
            double bias_mean = bias_sum / lane_bias_history_ptr->size();

            //// If current slope is close to slope mean
            // if( std::abs( slope - slope_mean ) < 0.1 )
            //{
            //    lane_slope_history_ptr->erase( lane_slope_history_ptr->begin() );
            //    lane_slope_history_ptr->push_back( slope );
            //    slope_sum = std::accumulate( lane_slope_history_ptr->begin(),
            //                                 lane_slope_history_ptr->end(), 0.0 );
            //    slope_mean = slope_sum / lane_slope_history_ptr->size();
            //}

            //// If current bias is close to bias mean
            // if( std::abs( bias - bias_mean ) < 10 )
            //{
            //    lane_bias_history_ptr->erase( lane_bias_history_ptr->begin() );
            //    lane_bias_history_ptr->push_back( bias );
            //    bias_sum = std::accumulate( lane_bias_history_ptr->begin(),
            //                                lane_bias_history_ptr->end(), 0.0 );
            //    bias_mean = bias_sum / lane_bias_history_ptr->size();
            //}

            /*for( auto bias : left_lane_bias_history )
            {
                cout << bias << ",";
                        }*/
            /*cout << endl;
            cout << "bias mean of history is: " << bias_mean << ", "
                 << "current bias mean is: " << bias << endl;
            cout << "slope mean of history is: " << slope_mean << ", "
                 << "current slope mean is: " << slope << endl;*/

			// y = kx+b, x = (y-b)/k
            if( lane_name == "r" )
            {
                right_lane_.at( 0 ).x = ( right_lane_.at( 0 ).y - bias_mean ) / slope_mean;
                right_lane_.at( 1 ).y = slope_mean * right_lane_.at( 1 ).x + bias_mean;
            }
            else
            {
                left_lane_.at( 0 ).x = ( left_lane_.at( 0 ).y - bias_mean ) / slope_mean;
                left_lane_.at( 1 ).y = slope_mean * left_lane_.at( 1 ).x + bias_mean;
            }
        }
    }
    // Stop Lane Detection
    // void isStopLanePresent( Mat img )
    //{
    //    std::vector<Vec4i> lines;
    //    cv::HoughLinesP( img, lines, 6, CV_PI / 60, 75, 40, 10 );

    //    vector<Point> stop_lines;
    //    // devide all detected lines into 2 groups accroding to slope
    //    for( const Vec4i& l : lines )
    //    {
    //        double slope = static_cast<double>( l[3] - l[1] ) / ( l[2] - l[0] );

    //        if( -0.1 < slope && slope < 0.1 )
    //        {
    //            stop_lines.emplace_back( l[0], l[1] );
    //            stop_lines.emplace_back( l[2], l[3] );
    //        }
    //    }

    //    vector<Point> stop_lane = {};
    //    if( !stop_lines.empty() )
    //    {
    //        is_stop_lane_present_ = true;
    //        stop_lane = mergeLines( original_image_, stop_lines, Scalar( 0, 255, 0 ), 10000.0,
    //                                original_image_.cols );
    //    }
    //    else
    //    {
    //        is_stop_lane_present_ = false;
    //        cout << "there is no stop lane in front of the car." << endl;
    //    }
    //}

    // Red Sign Detection
    void isRedLightPresent( cv::Mat img )
    {
        cv::Mat image8U;
        cv::Rect region( 0, 0, img.cols, img.rows * 0.55 );
        cv::Mat img_region = img( region );

        if( img_region.empty() )
        {
            std::cerr << "empty \n";
            return;
        }

        // if( img_region.type() != CV_8UC3 )
        //{
        //    // transform
        //    img_region.convertTo( image8U, CV_8UC3, 255.0 );
        //}
        // else
        //{
        //    image8U = img_region.clone();
        //}

        // from BGR to HSV
        cv::Mat hsv_img;
        try
        {
            cv::cvtColor( img_region, hsv_img, cv::COLOR_BGR2HSV );
        }
        catch( cv::Exception& e )
        {
            std::cerr << "error with cvtColor: " << e.what() << '\n';
            return;
        }

        // range of red
        cv::Scalar lower_red = cv::Scalar( 130, 90, 0 );
        cv::Scalar upper_red = cv::Scalar( 255, 255, 255 );

        // detect red color
        cv::Mat mask_red;
        cv::inRange( hsv_img, lower_red, upper_red, mask_red );
        // imshow( "mask red", mask_red );

        // how many pixels are in red
        int red_pixels = cv::countNonZero( mask_red );
        cout << "red_pixels =" << red_pixels << endl;
        // if more than 0 pixels are in green, brake
        if( red_pixels > 75 )
        {
            // cv::imshow( "Red Mask", mask_red );
            // cv::waitKey( 0 );  // 0 means wait for any key press
            is_red_sign_present_ = true;
            return;
        }
        else
        {
            is_red_sign_present_ = false;

            return;
        }
    }

    /* Aufgabe3: Steering Control     */
    void steeringControl( Mat img )
    {
        if( left_lane_.empty() || right_lane_.empty() )
        {
            steeringControlForMissingLanes( img );
            return;
        }
        // Use PID controller to control steering
        double target_point = ( left_lane_.front().x + right_lane_.front().x ) / 2;
        double curr_point = img.cols / 2;
        Point middle_point( ( left_lane_.front().x + right_lane_.front().x ) / 2,
                            left_lane_.front().y );
        int height = img.rows;

        //// draw the target driving direction
        // circle( img, middle_point, 1, Scalar( 255, 0, 0 ) );
        // line( img, Point( target_point, height - 1 ), middle_point, Scalar( 0, 255, 255 ), 5 );
        // line( img, Point( curr_point, height - 1 ), Point( target_point, height - 1 ),
        //      Scalar( 0, 0, 255 ), 10 );

        double steer_error_P =
            target_point - curr_point;  // current steering error = the difference between target
                                        // position and current position
        double steer_error_D =
            steer_error_P - steer_error_old_;  // The rate of change of the error = the difference
                                               // between current steering error and before
        steer_error_old_ = steer_error_P;
        steer_error_I_ = steer_error_I_ + steer_error_P;
        double steer_output = steer_error_P * steer_P_ + steer_error_D * steer_D_ +
                              steer_I_ * steer_error_I_;  // The output from PD controller

        cout << "Steering error:" << steer_error_P << ", " << steer_error_I_ << ", " << steer_error_D
             << "Steering output = " << steer_output_norm_ << endl;
        saveToCSV_PID( steer_error_P, steer_error_I_, steer_error_D, steer_output / ( img.cols / 2 ));
        // normalize the output between -1 and 1
        steer_output_norm_ = steer_output / ( img.cols / 2 );

        if( steer_output_norm_ > -0.05 && steer_output_norm_ < 0.05 )
        {
            steer_output_norm_ = 0;
        }
    }

    void steeringControlForMissingLanes( cv::Mat img )
    {
        int height = img.rows;
        double P = 7 / 1e3;
        double I = 3 / 1e5;
        double D = 0;
        double steer_error_P;

        if( left_lane_.empty() )
        {
            steer_error_P = 400 - right_lane_.at( 1 ).y;
        }
        else if( right_lane_.empty() )
        {
            steer_error_P = left_lane_.at( 1 ).y - 400;
        }
        double steer_error_D =
            steer_error_P - steer_error_old_;  // The rate of change of the error = the difference
                                               // between current steering error and before
        steer_error_old_ = steer_error_P;
        steer_error_I_ = steer_error_I_ + steer_error_P;
        double steer_output =
            steer_error_P * P + steer_error_D * D + steer_I_ * I;  // The output from PD controller

        cout << "Steering error:" << steer_error_P << ", " << steer_error_I_ << ", " << steer_error_D
             << "Steering output = " << steer_output << endl;
        steer_output_norm_ = steer_output;
        if( steer_output > 1 )
        {
            steer_output_norm_ = 1;
        }
        else if( steer_output < -1 )
        {
            steer_output_norm_ = -1;
        }

        if( steer_output_norm_ > -0.05 && steer_output_norm_ < 0.05 )
        {
            steer_output_norm_ = 0;
        }
    }
    /* Aufgabe4: throttle control*/
    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = ( msg->Velocity ) * 0.036;
        return true;
    }

    bool processBoundingBox( tronis::BoxDataSub* sensorData )
    {
        double min_distance = std::numeric_limits<double>::infinity();
        double distance;

        // Sensor may detect several objects, all objects should be processed
        for( size_t i = 0; i < sensorData->Objects.size(); i++ )
        {
            tronis::ObjectSub& object = sensorData->Objects[i];

            tronis::LocationSub location = object.Pose.Location;
            float pos_x = location.X;
            float pos_y = location.Y;

            tronis::ExtendSub extends = object.BB.Extends;
            float length = extends.X;
            float width = extends.Y;
            float height = extends.Z;

            string vehicle_name = object.ActorName.Value();

            // filter detected cars size
            distance = sqrt( pow( pos_x / 100, 2 ) + pow( pos_y / 100, 2 ) );
            /*std::cout << "The distance from" << object.ActorName.Value() << " is" << distance
                      << std::endl;
            cout << "length of object is:" << length << ", width is " << width << endl;
            cout << "pox x = " << pos_x << ","
                 << "pox y = " << pos_y << endl;*/
            if( length > 100 && length < 800 && width > 100 && width < 800 )
            {
                // Filter self car
                if( pos_x != 0.0 )
                {
                    // filter detected cars which in other lane
                    distance = sqrt( pow( pos_x / 100, 2 ) + pow( pos_y / 100, 2 ) );
                    /*  cout << " rightLaneVehicle.count( " << vehicle_name
                           << " )= " << rightLaneVehicle.count( vehicle_name )
                           << " leftLaneVehicle.count( " << vehicle_name
                           << " )= " << leftLaneVehicle.count( vehicle_name ) << "pos_y = " << pos_y
                           << endl;*/

                    // if it's first time to detect this car
                    if( rightLaneVehicle.count( vehicle_name ) == 0 &&
                        leftLaneVehicle.count( vehicle_name ) == 0 )
                    {
                        if( std::abs( pos_y ) < 400.0 )
                        {
                            rightLaneVehicle[vehicle_name] = distance;
                            distance_ = distance;

                            /* cout << vehicle_name
                                  << "is in right lane, distance from it = " << distance << endl;*/
                        }
                        else
                        {
                            leftLaneVehicle[vehicle_name] = distance;
                            /*cout << vehicle_name
                                 << "is in left lane, distance from it = " << distance << endl;*/
                        }
                    }
                    // The car has been detected
                    else
                    {
                        if( rightLaneVehicle.count( vehicle_name ) > 0 )
                        {
                            rightLaneVehicle[vehicle_name] = distance;
                            distance_ = distance;

                            /*cout << vehicle_name
                                 << "is in right lane, distance from it = " << distance << endl;*/
                        }
                        else
                        {
                            leftLaneVehicle[vehicle_name] = distance;
                            /*cout << vehicle_name
                                 << "is in left lane, distance from it = " << distance << endl;*/
                        }
                    }
                }
            }
        }

        if( distance_ >= max_detected_distance_ )
        {
            distance_ = std::numeric_limits<double>::infinity();
            isCardAhead = false;
        }
        else
        {
            isCardAhead = true;
        }
        return true;
    }

    void throttleControl()
    {
        cout << "isStopLanePresent: " << is_stop_lane_present_ << ", "
             << "isRedLightPresent: " << is_red_sign_present_ << ","
             << "isStop:" << isStopHandled_ << endl;

        // Brake Assistant
        if( is_red_sign_present_ == true && is_stop_lane_present_ == true && !isStopHandled_ )
        {
            // cout << "red is present, shoule brake!" << endl;

            if( ego_velocity_ < 1 )  // check if the car stop already or not
            {
                stopCarForDuration( std::chrono::seconds( 3 ) );
                isStopHandled_ = true;
            }
            else
            {
                throttle_output_norm_ = 0;
            }

            return;
        }
        else if( isStopHandled_ )
        {
            currentTime_ = std::chrono::high_resolution_clock::now();
            if( currentTime_ - startTime_ > std::chrono::seconds( 5 ) )
            {
                isStopHandled_ = false;
            }
        }

        // Throttle Control
        // if( isCardAhead && ( distance_ < min_distance + 5 && ego_velocity_ > 40 ) ||
        //    ( distance_ < min_distance + 10 && ego_velocity_ > 45 ) )
        //{
        //    throttle_output_norm_ = -0.5;
        //}
        // else if( isCardAhead && distance_ <= min_distance )
        //{
        //    if( abs( ego_velocity_ ) < 1 )  // make it absolutely stop
        //    {
        //        throttle_output_norm_ = 0;
        //    }
        //    else  // urgent stop
        //    {
        //        throttle_output_norm_ = -0.5;
        //    }
        //}
        // else
        //{
        //    if( abs( ego_velocity_ ) > max_velocity )  // stop accerating before a sharp turn
        //    {
        //        throttle_output_norm_ = -0.5;
        //    }

        //    if( abs( ego_velocity_ ) > 20 )  // to keep the cvelocity stable
        //    {
        //        speedControl( max_velocity, 0 );
        //    }

        //    else  // make it reaccelerate faster after deaccelerating
        //    {
        //        throttle_output_norm_ = 1;
        //    }
        //}

        if( isCardAhead )
        {
            distanceControl();
        }
        else
        {
            speedControl( max_velocity );
        }
    }

    void speedControl( double target_velocity )
    {
        double throttle_error_P = target_velocity - ego_velocity_;
        throttle_error_I = throttle_error_I + throttle_error_old;
        double throttle_error_D = throttle_error_old - throttle_error_P;

        throttle_error_old = throttle_error_P;

        double throttle_output = throttle_error_P * throttle_P + throttle_error_D * throttle_D +
                                 throttle_error_I * throttle_I;

        throttle_output_norm_ = std::min( throttle_output, 1.0 );

        // Avoid over maximum velocity
        if( ego_velocity_ > max_velocity && !isCardAhead )
        {
            throttle_output_norm_ = 0.6;
        }

        cout << "Speed PID is P = " << throttle_error_P << ", I = " << throttle_error_I
             << ", D =" << throttle_error_D << endl;
        cout << "Throttle output = " << throttle_output_norm_ << endl;
    }

    void distanceControl()
    {
        double distance_target = 0.7 * ego_velocity_;

        if( ego_velocity_ < 20 )
        {
            distance_target = 20;
        }

        double distance_error_P = distance_ - distance_target;
        double distance_error_D = distance_error_P - distance_error_old;
        distance_error_I = distance_error_I + distance_error_old;

        distance_error_old = distance_error_P;

        double velocity_target = ego_velocity_ + distance_error_P * distance_P +
                                 distance_error_D * distance_D + distance_error_I * distance_I;

        // Avoid negative velocity
        if( velocity_target < 1 )
        {
            velocity_target = 0;
        }
        else if( velocity_target > max_velocity )
        {
            velocity_target = max_velocity;
        }

        speedControl( velocity_target );
        cout << "distance PID Controller: P = " << distance_error_P << ", I = " << distance_error_I
             << ", D = " << distance_error_D << endl;
        cout << "Velocity Target is " << velocity_target << ", ego velocity = " << ego_velocity_
             << ", distance = " << distance_ << endl;
    }

    void stopCarForDuration( std::chrono::seconds duration )
    {
        startTime_ = std::chrono::high_resolution_clock ::now();

        while( true )
        {
            throttle_output_norm_ = 0;
            currentTime_ = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = currentTime_ - startTime_;
            if( diff > duration )
            {
                cout << "diff is: " << diff.count() << endl;
                break;
            }
        }
    }

    void saveToCSV( vector<Point> lane, bool isRightLane )
    {
        std::ofstream lines_data;
        lines_data.open( "D:\\200_Projekte\\distanceData.csv", ios::app );

        double slope = static_cast<double>( lane.at( 1 ).y - lane.at( 0 ).y ) /
                       ( lane.at( 1 ).x - lane.at( 0 ).x );
        if( isRightLane )
        {
            lines_data << slope << endl;
        }
        else
        {
            lines_data << ""
                       << "," << slope << endl;
        }

        lines_data.close();
    }

	 void saveToCSV_PID( double P,double I,double D, double output )
    {
        std::ofstream lines_data;
        lines_data.open( "D:\\200_Projekte\\distanceData.csv", ios::app );

        lines_data << P << "," << I << "," << D << "," << output << endl;

        lines_data.close();
    }
    std::string formatDouble( double value )
    {
        std::ostringstream streamObj;
        streamObj << std::fixed;
        streamObj << std::setprecision( 2 );
        streamObj << value;
        return streamObj.str();
    }

public:
    void initialCSV()
    {
        std::ofstream distanceData;
        distanceData.open( "D:\\200_Projekte\\distanceData.csv", std::ios::trunc );
        /*distanceData << "Slope"
                     << ","
                     << "Bias" << endl;*/
        distanceData << "P"
                     << ","
                     << "I"
                     << ","
                     << "D"
                     << ","
                     << "output" << endl;
        distanceData.close();
    }
    // Helper functions, no changes needed
public:
    // Function to process received tronis data
    bool getData( tronis::ModelDataWrapper data_model )
    {
        if( data_model->GetModelType() == tronis::ModelType::Tronis )
        {
            std::cout << "Id: " << data_model->GetTypeId() << ", Name: " << data_model->GetName()
                      << ", Time: " << data_model->GetTime() << std::endl;

            // if data is sensor output, process data
            switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
            {
                case tronis::TronisDataType::Image:
                {
                    processImage( data_model->GetName(),
                                  data_model.get_typed<tronis::ImageSub>()->Image );
                    break;
                }
                case tronis::TronisDataType::ImageFrame:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFrameSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::ImageFramePose:
                {
                    const tronis::ImageFrame& frames(
                        data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                    for( size_t i = 0; i != frames.numImages(); ++i )
                    {
                        std::ostringstream os;
                        os << data_model->GetName() << "_" << i + 1;

                        processImage( os.str(), frames.image( i ) );
                    }
                    break;
                }
                case tronis::TronisDataType::PoseVelocity:
                {
                    processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                    break;
                }
                case tronis::TronisDataType::BoxData:
                {
                    processBoundingBox( data_model.get_typed<tronis::BoxDataSub>() );
                    break;
                }
                default:
                {
                    std::cout << data_model->ToString() << std::endl;
                    break;
                }
            }
            return true;
        }
        else
        {
            std::cout << data_model->ToString() << std::endl;
            return false;
        }
    }

protected:
    // Function to show an openCV image in a separate window
    void showImage( std::string image_name, cv::Mat image )
    {
        cv::Mat out = image;
        if( image.type() == CV_32F || image.type() == CV_64F )
        {
            cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
        }
        cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );
        cv::imshow( image_name.c_str(), out );
    }

    // Function to convert tronis image to openCV image
    bool processImage( const std::string& base_name, const tronis::Image& image )
    {
        cout << " Start to process image. " << endl;
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_ = tronis::image2Mat( image );
        original_image_ = image_;

        detectLanes();
        processed_frames += 1;
    }
};

// main loop opens socket and listens for incoming data

int main( int argc, char** argv )
{
    std::ofstream logout( " logout.txt " );
    std::cout.rdbuf( logout.rdbuf() );
    std::cout << "Welcome to lane assistant" << std::endl;

    // specify socket parameters
    std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "50542";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
                  << "\", PortBind:" << socket_port << "}";

    int key_press = 0;  // close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param

    LaneAssistant lane_assistant;
    lane_assistant.initialCSV();

    while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );
        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
        tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
            // wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
                // data received! reset timer
                time_ms = 0;

                // convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
                // identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
                // no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}

// int main( int argc, char** argv )
//{
//    Mat original_img = cv::imread( "C:\\Users\\am3s33\\Pictures\\Camera Roll\\test1.png" );
//
//    LaneAssistant laneassistant;
//    laneassistant.processImage( original_img );
//    waitKey( 0 );
//}
