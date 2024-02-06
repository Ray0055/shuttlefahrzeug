#include <iostream>
#include <fstream>
#include <string>
#include <random>
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
                             std::to_string( isCardAhead ) + ";" + formatDouble( distance_ ) + ";" +
                             formatDouble( velocity_front_car_ );
        socket.send( tronis::SocketData( control_cmd ) );

        return true;
    }

    void processImage( cv::Mat image )
    {
        image_ = image;
        original_image_ = image;
        regionDetection( image_ );

        colorDetection( image_ );

        edgeDetection( image_ );

        lineDetection( image_ );

        BEV_image_ = birdEyeView( image_, true );

        steeringControl( original_image_ );
        showImage( "original image", original_image_ );
        showImage( "BEV_image_", BEV_image_ );
        waitKey( 0 );
    }

    void processImage_2( Mat img )
    {
        image_ = img;
        original_image_ = img;
        regionDetection( image_ );
        Mat BEV_img_color = birdEyeView_fullscreen( image_, true );

        colorDetection( image_ );

        BEV_image_ = birdEyeView_fullscreen( image_, true );

        if( right_lane.detected && left_lane.detected )
        {
            line_detection_by_previous( BEV_image_, BEV_img_color, 9, left_lane, right_lane );
        }
        else
        {
            line_dection_by_sliding_window( BEV_image_, BEV_img_color, 9, left_lane, right_lane );
        }

        draw_detected_lane_onto_road( original_image_, left_lane, right_lane );
        // get_center_of_road( original_image_, BEV_img_color, left_lane, right_lane );
        steeringControl_poly( BEV_img_color );
        cv::putText( BEV_img_color, "Steering:" + to_string( steer_output_norm_ ), Point( 300, 45 ),
                     FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 255, 0 ), 1 );
        imshow( "BEV_img_color", BEV_img_color );
        imshow( "output", original_image_ );
        waitKey( 0 );
    }

    void stopLineDetectionMethod( Mat img )
    {
        image_ = img;
        original_image_ = img;
        isRedLightPresent( image_ );
        regionDetection( image_ );

        Mat img_color_detected = colorDetection( image_ );

        edgeDetection( img_color_detected );

        isStopLanePresent( image_ );

        cout << "isStopLanePresent: " << stop_lane_ << ", "
             << "isRedLightPresent: " << red_stop_sign_ << endl;
    }

protected:
    std::string image_name_;
    cv::Mat image_, original_image_, BEV_image_;
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;
    vector<Point> left_lane_, right_lane_;
    double steer_output_norm_, throttle_output_norm_;
    int processed_frames = 0;
    Lane right_lane, left_lane;

    // Stop Sign and Stop Lane Detection
    bool red_stop_sign_ = false;
    bool stop_lane_ = false;
    bool isBrake_ = false;  // need to brake or not
    bool isStop_ = false;   // check if the car has stopped for duration
    bool isWaiting = false;
    // hyperparameter of PID controller
    double steer_P_ = 2.0;
    double steer_D_ = 0.0000;
    double steer_I_ = 0.0000;

    // Initialize error, derivative of error, integration of error
    double steer_error_old_ = 0;
    double steer_error_I_ = 0;

    // Initialize throttle PID controller and parameters

    double max_velocity = 50;  // max velocity
    double throttle_P = 0.2;
    double throttle_I = -0.00001;
    double throttle_D = -0.02;
    double throttle_error_old = 0;
    double throttle_error_I = 0;

    // Initialize distance PID controller and parameters
    bool isCardAhead;
    double distance_;
    double velocity_front_car_;
    double old_distance_;
    double min_distance = 30;  // min safe car-car distance
    double distance_P = 0.03;
    double distance_I = 0;
    double distance_D = 0.0003;
    double distance_error_old = 0;
    double distance_error_I = 0;

    double new_time;
    double old_time;
    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        regionDetection( image_ );

        colorDetection( image_ );

        edgeDetection( image_ );

        lineDetection( image_ );

        BEV_image_ = birdEyeView( image_, true );

        steeringControl( original_image_ );
        throttle_output_norm_ = 0.0;
    }

    void detectLanes_poly()
    {

        //isRedLightPresent( image_ );

        regionDetection( image_ );
        Mat BEV_img_color = birdEyeView_fullscreen( image_, true );

        Mat image_color_detected = colorDetection( image_ );
        cv::imshow( "image_color_detected", image_color_detected );

        //edgeDetection( image_color_detected );
        // isStopLanePresent( image_ );
        stop_lane_ = true;
        Mat edge_detected_image = edgeDetectionPoly( image_color_detected );
        BEV_image_ = birdEyeView_fullscreen( edge_detected_image, true );
        cv::imshow( "BEV_image", BEV_image_ );
        if( right_lane.detected && left_lane.detected )
        {
            line_detection_by_previous( BEV_image_, BEV_img_color, 9, left_lane, right_lane );
        }
        else
        {
            line_dection_by_sliding_window( BEV_image_, BEV_img_color, 9, left_lane, right_lane );
        }

        if( !right_lane.last_lane_points_pixel.empty() &&
            !left_lane.last_lane_points_pixel.empty() )
        {
            draw_detected_lane_onto_road( original_image_, left_lane, right_lane );
        }

        steeringControl_poly( BEV_img_color );

        cv::polylines( BEV_img_color, right_lane.last_center_pts_pixel, false,
                       Scalar( 246, 161, 75 ), 5 );
        
		BEV_image_ = BEV_img_color;

        throttleControl();
        // throttle_output_norm_ = 0.0;
        if( isCardAhead )
        {
            velocity_front_car_ = speedEstimator();
            /*cv::putText(
                BEV_img_color,
                "Velocity of the front car:" + to_string( std::round( speed * 100 ) / 100 ),
                Point( 300, 245 ), FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 255, 0 ), 1 );

                        cv::putText( BEV_img_color,
                         "Distance:" + to_string( std::round( distance_ * 100 ) / 100 ),
                         Point( 300, 145 ), FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 255, 0 ), 1 );*/
        }
    }

    void regionDetection( Mat img )
    {
        /*Create a region of interest to concentrate on the road*/

        cout << "Start to detect region: " << endl;
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
        pts.push_back( Point( img_width, img_height ) );           // bottom left
        pts.push_back( Point( 0, img_height ) );                   // bottom right

        v_pts.push_back( pts );

        /// add cone to mask
        fillPoly( mask_crop, v_pts, ( 255, 255, 255 ) );

        if( img.type() == CV_8UC3 )
        {
            Mat mask_three_channel;
            cv::cvtColor( mask_crop, mask_three_channel,
                          COLOR_GRAY2BGR );  // transform mask into three channels
            cv::bitwise_and( img, mask_three_channel, result );
            cout << " Region is detected." << endl;
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
            cout << " Region is detected." << endl;
        }

        image_ = result;
    }
    Mat colorDetection( cv::Mat img )
    {
        /*According to HSV color space, this function will pick white and yellow region from
         * original region*/
        Mat mask_yellow, mask_white, mask_combined, img_hsv;
        // Transform image from RBG to HSV color space
        cvtColor( img, img_hsv, COLOR_BGR2HSV );

        // yellow region hsv threshold
        cv::Scalar lower_yellow( 20, 100, 120 );
        cv::Scalar upper_yellow( 30, 255, 255 );
        cv::inRange( img_hsv, lower_yellow, upper_yellow, mask_yellow );

        // white region hsv threshold
        cv::Scalar lower_white( 106, 0, 130 );
        cv::Scalar upper_white( 173, 90, 200 );
        cv::inRange( img_hsv, lower_white, upper_white, mask_white );

        // Combine two regions
        cv::bitwise_or( mask_yellow, mask_white, mask_combined );
        cout << " Color is detected." << endl;
        return mask_combined;
        cv::imshow( "Color Detection", mask_combined );
    }
    void edgeDetection( cv::Mat image )
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
        cv::GaussianBlur( image, blurred_img, cv::Size( 3, 3 ), 20, 0, cv::BORDER_DEFAULT );

        cv::Canny( image, grad_img, 50, 200 );

        cout << " Edge is detected." << endl;
        image_ = grad_img;
    }
    cv::Mat edgeDetectionPoly( cv::Mat image )
    {
        cv::Mat grad_img, blurred_img;

        // Canny Edge Detector
        cv::GaussianBlur( image, blurred_img, cv::Size( 3, 3 ), 20, 0, cv::BORDER_DEFAULT );

        cv::Canny( image, grad_img, 50, 200 );

        return grad_img;
    }
    Mat birdEyeView( Mat img, bool mode )
    {
        /*Create bird eye view*/

        int image_height = img.size().height;
        int image_width = img.size().width;

        vector<Point2f> src( 4 ), dst( 4 );
        src[0] = Point2f( 0, image_height / 2 + 45 );            // top left
        src[1] = Point2f( 0, image_height );                     // bottom left
        src[2] = Point2f( image_width, image_height );           // bottom right
        src[3] = Point2f( image_width, image_height / 2 + 45 );  // top right

        dst[0] = Point2f( 0, 0 );                                // top left
        dst[1] = Point2f( image_width / 2 - 45, image_height );  // bottom left
        dst[2] = Point2f( image_width / 2 + 45, image_height );  // bottom right
        dst[3] = Point2f( image_width, 0 );                      // top right

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
    void lineDetection( cv::Mat img )
    {
        std::vector<Vec4i> lines;
        cv::HoughLinesP( img, lines, 6, CV_PI / 60, 75, 40, 10 );

        if( lines.empty() )
        {
            cout << "no line is detected!" << endl;
        }

        // Mat img_color;
        //     cvtColor( img, img_color, COLOR_GRAY2BGR );
        //     // painting lines in original image
        //      for( size_t i = 0; i < lines.size(); i++ )
        //     {
        //         Vec4i l = lines[i];
        //
        //         line( img_color, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ),
        //         3,
        //               LINE_AA );
        //     }

        //    imshow( "dada",img_color );
    }
    void drawLans_line( Mat img, vector<Vec4i> lines )
    {
        Mat img_color;
        vector<Point> left_points_, right_points_;  // the points of left and right lanes
        cvtColor( img, img_color, COLOR_GRAY2BGR );
        // devide all detected lines into 2 groups accroding to slope
        for( const Vec4i& l : lines )
        {
            double slope = static_cast<double>( l[3] - l[1] ) / ( l[2] - l[0] );

            if( slope > 0 )
            {
                right_points_.emplace_back( l[0], l[1] );
                right_points_.emplace_back( l[2], l[3] );
            }
            else
            {
                left_points_.emplace_back( l[0], l[1] );
                left_points_.emplace_back( l[2], l[3] );
            }
        }

        int imgHeight = img_color.rows;
        // merge and draw left lane
        if( !left_points_.empty() )
        {
            left_lane_ = mergeLines( original_image_, left_points_, Scalar( 255, 0, 0 ), 10000.0,
                                     imgHeight );
        }

        // merge and draw right lane
        if( !right_points_.empty() )
        {
            right_lane_ = mergeLines( original_image_, right_points_, Scalar( 0, 255, 0 ), 10000.0,
                                      imgHeight );
        }

        cout << "Lines are detected" << endl;
    }

    /*Use fitLine() function to merge all line which point to same direction to one line*/
    vector<Point> mergeLines( Mat& image, const vector<Point>& points, Scalar color,
                              double limitedLength, int img_height )
    {
        Vec4f line_parameters;
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
        point2.x = point1.x + ( img_height - point1.y ) / k;
        point2.y = img_height;

        //
        line( image, point1, point2, color, 2, LINE_AA );

        // Save all the lane points
        vector<Point> points_lane;
        for( int i = img_height / 2 + 40; i < img_height; i++ )
        {
            points_lane.push_back(
                Point( point1.x + ( i - point1.y ) / line_parameters[1] * line_parameters[0], i ) );
        }

        return points_lane;
    }

    /* Stop Lane Detection*/
    void isStopLanePresent( Mat img )
    {
        std::vector<Vec4i> lines;
        cv::HoughLinesP( img, lines, 6, CV_PI / 60, 75, 40, 10 );

        vector<Point> stop_lines;
        // devide all detected lines into 2 groups accroding to slope
        for( const Vec4i& l : lines )
        {
            double slope = static_cast<double>( l[3] - l[1] ) / ( l[2] - l[0] );

            if( -0.1 < slope && slope < 0.1 )
            {
                stop_lines.emplace_back( l[0], l[1] );
                stop_lines.emplace_back( l[2], l[3] );
            }
        }

        vector<Point> stop_lane = {};
        if( !stop_lines.empty() )
        {
            stop_lane_ = true;
            stop_lane = mergeLines( original_image_, stop_lines, Scalar( 0, 255, 0 ), 10000.0,
                                    original_image_.cols );
        }
        else
        {
            stop_lane_ = false;
            cout << "there is no stop lane in front of the car." << endl;
        }
    }

    /* Red Sign Detection*/
    void isRedLightPresent( cv::Mat image_ )
    {
        cv::Mat image8U;
        cv::Rect region( 0, 0, image_.cols, image_.rows * 0.55 );
        cv::Mat img_region = image_( region );
        if( img_region.empty() )
        {
            std::cerr << "empty \n";
            return;
        }

        if( img_region.type() != CV_8UC3 )
        {
            // transform
            img_region.convertTo( image8U, CV_8UC3, 255.0 );
        }
        else
        {
            image8U = img_region.clone();
        }

        // from BGR to HSV
        cv::Mat hsv_img;
        try
        {
            cv::cvtColor( image8U, hsv_img, cv::COLOR_BGR2HSV );
        }
        catch( cv::Exception& e )
        {
            std::cerr << "error with cvtColor: " << e.what() << '\n';
            return;
        }

        // range of red
        cv::Scalar lower_red = cv::Scalar( 135, 90, 0 );
        cv::Scalar upper_red = cv::Scalar( 255, 255, 255 );
        // detect red color
        cv::Mat mask_red;
        cv::inRange( hsv_img, lower_red, upper_red, mask_red );
        imshow( "mask red", mask_red );

        // how many pixels are in red
        int red_pixels = cv::countNonZero( mask_red );
        cout << "red_pixels =" << red_pixels << endl;
        // if more than 0 pixels are in green, brake
        if( red_pixels > 0 )
        {
            // cv::imshow( "Red Mask", mask_red );
            // cv::waitKey( 0 );  // 0 means wait for any key press
            red_stop_sign_ = true;
            return;
        }
        else
        {
            red_stop_sign_ = false;
            return;
        }
    }

    /*     Steering       */
    void steeringControl( Mat img )
    {
        if( left_lane_.empty() || right_lane_.empty() )
        {
            return;
        }
        // Use PID controller to control steering
        double target_point = ( left_lane_.back().x + right_lane_.back().x ) / 2;
        double curr_point = img.cols / 2;
        Point middle_point( ( left_lane_.front().x + right_lane_.front().x ) / 2,
                            left_lane_.front().y );
        int height = original_image_.rows;

        // draw the target driving direction
        circle( original_image_, middle_point, 1, Scalar( 255, 0, 0 ) );
        line( original_image_, Point( target_point, height - 1 ), middle_point,
              Scalar( 0, 255, 255 ), 5 );
        line( original_image_, Point( curr_point, height - 1 ), Point( target_point, height - 1 ),
              Scalar( 0, 0, 255 ), 10 );

        double steer_error_P =
            target_point - curr_point;  // current steering error = the difference between target
                                        // position and current position
        double steer_erro_D =
            steer_error_P - steer_error_old_;  // The rate of change of the error = the difference
                                               // between current steering error and before
        steer_error_old_ = steer_error_P;
        steer_error_I_ = steer_error_I_ + steer_error_P;
        double steer_output = steer_error_P * steer_P_ + steer_erro_D * steer_D_ +
                              steer_I_ * steer_error_I_;  // The output from PD controller

        // normalize the output between -1 and 1
        steer_output_norm_ = steer_output / ( original_image_.cols / 2 );

        cv::putText( original_image_, "Steering:" + to_string( steer_output_norm_ ),
                     Point( 300, 45 ), FONT_HERSHEY_COMPLEX, 1, Scalar( 0, 255, 0 ), 1 );
    }

    void steeringControl_poly( Mat BEV_img_color )
    {
        vector<Point> center_lane_pts_pixel = right_lane.last_center_pts_pixel;
        if( center_lane_pts_pixel.empty() )
        {
            return;
        }

        // Use PID controller to control steering
        Point middle_point = center_lane_pts_pixel.back();
        double target_point = middle_point.x;
        double curr_point = BEV_img_color.cols / 2;
        int height = BEV_img_color.rows;

        // draw the target driving direction
        line( BEV_img_color, Point( curr_point, height - 1 ), middle_point, Scalar( 0, 0, 255 ),
              5 );

        double steer_error_P =
            target_point - curr_point;  // current steering error = the difference between target
                                        // position and current position
        double steer_erro_D =
            steer_error_P - steer_error_old_;  // The rate of change of the error = the difference
                                               // between current steering error and before
        steer_error_old_ = steer_error_P;
        steer_error_I_ = steer_error_I_ + steer_error_P;
        double steer_output = steer_error_P * steer_P_ + steer_erro_D * steer_D_ +
                              steer_I_ * steer_error_I_;  // The output from PD controller

        // normalize the output between -1 and 1
        steer_output_norm_ = steer_output / ( 400 / 2.0 );

        if( steer_output_norm_ < -1.0 )
        {
            steer_output_norm_ = -1.0;
        }
        else if( steer_output_norm_ > 1.0 )
        {
            steer_output_norm_ = 1.0;
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

            // filter detected cars
            if( length > 100 && length < 800 && width > 100 && width < 800 )
            {
                distance = sqrt( pow( pos_x / 100, 2 ) + pow( pos_y / 100, 2 ) );
                if( distance < min_distance && distance > 0 )
                {
                    min_distance = distance;
                    std::cout << "The distance from" << object.ActorName.Value() << " is"
                              << min_distance << std::endl;
                    cout << "length of object is:" << length << ", width is " << width << endl;
                }
            }
        }
        old_distance_ = distance_;
        distance_ = min_distance;

        if( distance_ == std::numeric_limits<double>::infinity() )
        {
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
        double velocity_compensation = distanceControl();
        double velocity_of_front_car = speedEstimator();

        cout << "isStopLanePresent: " << stop_lane_ << ", "
             << "isRedLightPresent: " << red_stop_sign_ << endl;

        
        if( red_stop_sign_ == true && stop_lane_ == true &&
            isStop_ == false 
          )  // detect red stop sign and car is not stopped
        {
          
            throttle_output_norm_ = -0.5;
            if( ego_velocity_ < 1 )  // check if the car stop already or not
            {
                stopCarForDuration( std::chrono::seconds( 3 ) );
            }
           
            return;
        }
        else if( red_stop_sign_ == true && stop_lane_ == true &&
                 isStop_ == true )  // detect red stop sign and car is  stopped
        {
            isBrake_ = false;
        }
        else
        {
            isBrake_ = false;
        }

        if( isCardAhead && ( distance_ < min_distance + 5 && ego_velocity_ > 40 ) ||
            ( distance_ < min_distance + 10 && ego_velocity_ > 45 ) )
        {
            throttle_output_norm_ = -0.5;
        }
        else if( isCardAhead && distance_ <= min_distance )
        {
            if( abs( ego_velocity_ ) < 1 )  // make it absolutely stop
            {
                throttle_output_norm_ = 0;
            }
            else  // urgent stop
            {
                throttle_output_norm_ = -0.5;
            }
        }
        else
        {
            if( abs( ego_velocity_ ) > max_velocity )  // stop accerating before a sharp turn
            {
                throttle_output_norm_ = -0.5;
            }

            if( abs( ego_velocity_ ) > 20 )  // to keep the cvelocity stable
            {
                speedControl( max_velocity, 0 );
            }

            else  // make it reaccelerate faster after deaccelerating
            {
                throttle_output_norm_ = 1;
            }
        }
    }

    void speedControl( double taget_velocity, double velocity_compensation )
    {
        double throttle_error_P = taget_velocity - ego_velocity_ + velocity_compensation;
        throttle_error_I = throttle_error_I + throttle_error_old;
        double throttle_error_D = throttle_error_old - throttle_error_P;

        throttle_error_old = throttle_error_P;

        double throttle_output = throttle_error_P * throttle_P + throttle_error_D * throttle_D +
                                 throttle_error_I * throttle_I;

        throttle_output_norm_ = std::min( throttle_output, 1.0 );

        cout << "with speed controller, current throttle output is:" << throttle_output_norm_
             << endl;
        cout << "PID is " << throttle_error_P << "," << throttle_error_I << "," << throttle_error_D
             << endl;
    }

    std::string formatDouble( double value )
    {
        std::ostringstream streamObj;
        streamObj << std::fixed;
        streamObj << std::setprecision( 2 );
        streamObj << value;
        return streamObj.str();
    }

    double distanceControl()
    {
        double distance_error_P = distance_ - min_distance;
        double distance_error_D = distance_error_P - distance_error_old;
        distance_error_I = distance_error_I + distance_error_old;

        distance_error_old = distance_error_P;

        double distance_output = distance_error_P * distance_P + distance_error_D * distance_D +
                                 distance_error_I * distance_I;

        return distance_output;
    }

    double speedEstimator()
    {
        double distance_gap = distance_ - old_distance_;
        double speed_ahead_car = ego_velocity_ + distance_gap / 0.016666;
        return speed_ahead_car;
    }

    void stopCarForDuration( std::chrono::seconds duration )
    {
        auto startTime = std::chrono::high_resolution_clock ::now();

		isWaiting = true;
        while( isWaiting )
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            if( currentTime - startTime >= duration )
            {
                isWaiting = false;  
				isBrake_ = false;
                isStop_ = true;
                }

			}
        throttle_output_norm_ = -1;
        
        
    }
    void saveToCSV()
    {
        std::ofstream distanceData;
        distanceData.open( "D:\\200_Projekte\\distanceData.csv", ios::app );
        if( distance_ < std::numeric_limits<double>::infinity() && distance_ < 60 )
        {
            distanceData << to_string( distance_ ) << "," << to_string( throttle_output_norm_ )
                         << endl;
        }
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
                    old_time = new_time;
                    new_time = data_model->GetTime();

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

        detectLanes_poly();

        showImage( "Lane detection", original_image_ );
        showImage( "BEV_image", BEV_image_ );
    }
};

// main loop opens socket and listens for incoming data

int main( int argc, char** argv )
{
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
//    Mat original_img = imread( "C:\\Users\\am3s33\\Pictures\\Camera Roll\\stop_test6.png" );
//
//    LaneAssistant laneassistant;
//    laneassistant.stopLineDetectionMethod( original_img );
//    waitKey( 0 );
//}
