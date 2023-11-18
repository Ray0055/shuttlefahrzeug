#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>

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
        // do stuff with data

        // send results via socket
        return true;
    }

    void processImage( cv::Mat image )
    {
        image_ = image;
        
        // imshow( "orignial image", image_ );

        regionDetection( image_ );
        // imshow( "rigion detection", image_ );
        birdEyeView( image_ );
        original_image_ = BEV_image_;
        colorDetection( image_ );
        // imshow( "color detection", image_ );

        // edgeDetection( image_ );
        // imshow( "edge detection", image_ );

        // lineDetection( image_ );
        // imshow( "lane detection", image_ );
        
        
        birdEyeView( image_ );
        // imshow( "bird eye view", BEV_image_ );
        lineDetection2( BEV_image_, 9 );
        waitKey( 0 );
    }

protected:
    std::string image_name_;
    cv::Mat image_, original_image_, BEV_image_;
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;

    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        cout << "Start to detect lines: " << endl;

        regionDetection( image_ );

        colorDetection( image_ );

        edgeDetection( image_ );

        lineDetection( image_ );

        birdEyeView( image_ );
    }

    bool processPoseVelocity( tronis::PoseVelocitySub* msg )
    {
        ego_location_ = msg->Location;
        ego_orientation_ = msg->Orientation;
        ego_velocity_ = msg->Velocity;
        return true;
    }

    bool processObject()
    {
        // do stuff
        return true;
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
    void colorDetection( cv::Mat img )
    {
        /*According to HSV color space, this function will pick white and yellow region from
         * original region*/
        Mat mask_yellow, mask_white, mask_combined, img_hsv;
        // Transform image from RBG to HSV color space
        cvtColor( img, img_hsv, COLOR_BGR2HSV );

        // yellow region hsv threshold
        cv::Scalar lower_yellow( 20, 100, 100 );
        cv::Scalar upper_yellow( 30, 255, 255 );
        cv::inRange( img_hsv, lower_yellow, upper_yellow, mask_yellow );

        // white region hsv threshold
        cv::Scalar lower_white( 106, 0, 150 );
        cv::Scalar upper_white( 173, 90, 230 );
        cv::inRange( img_hsv, lower_white, upper_white, mask_white );

        // Combine two regions
        cv::bitwise_or( mask_yellow, mask_white, mask_combined );
        cout << " Color is detected." << endl;
        image_ = mask_combined;
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
    void birdEyeView( Mat img )
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
        dst[1] = Point2f( image_width / 2 - 90, image_height );  // bottom left
        dst[2] = Point2f( image_width / 2 + 90, image_height );  // bottom right
        dst[3] = Point2f( image_width, 0 );                      // top right

        Mat M = getPerspectiveTransform( src, dst );
        Mat Minv = getPerspectiveTransform( dst, src );
        Mat BEV_img;

        warpPerspective( img, BEV_img, M, Size( image_width, image_height ) );

        BEV_image_ = BEV_img;
    }
    void lineDetection( cv::Mat img )
    {
        Mat img_color;
        std::vector<Vec4i> lines;
        cv::HoughLinesP( img, lines, 6, CV_PI / 60, 75, 40, 10 );

        cvtColor( img, img_color, COLOR_GRAY2BGR );
        if( lines.empty() )
        {
            cout << "no line is detected!" << endl;
        }
        //// painting lines in original image
        // for( size_t i = 0; i < lines.size(); i++ )
        //{
        //    Vec4i l = lines[i];
        //
        //    line( img_color, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3,
        //          LINE_AA );
        //}

        // image_ = img_color;

        // devide all detect lines into 2 groups accroding to slope
        vector<Point> left_points, right_points;

        for( const Vec4i& l : lines )
        {
            double slope = static_cast<double>( l[3] - l[1] ) / ( l[2] - l[0] );

            if( slope > 0 )
            {
                right_points.emplace_back( l[0], l[1] );
                right_points.emplace_back( l[2], l[3] );
            }
            else
            {
                left_points.emplace_back( l[0], l[1] );
                left_points.emplace_back( l[2], l[3] );
            }
        }

        image_ = original_image_;
        int imgHeight = img_color.rows;
        // merge and draw left lane
        if( !left_points.empty() )
        {
            mergeLines( original_image_, left_points, Scalar( 255, 0, 0 ), 10000.0, imgHeight );
        }

        // merge and draw right lane
        if( !right_points.empty() )
        {
            mergeLines( original_image_, right_points, Scalar( 0, 255, 0 ), 10000.0, imgHeight );
        }

        cout << "Lines are detected" << endl;
    }

    void mergeLines( Mat& image, const vector<Point>& points, Scalar color, double limitedLength,
                     int img_height )
    {
        /*Use fitLine() function to merge all line which point to same direction to one line*/

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
    }

    void lineDetection2( Mat img, int number_of_windows )
    {
        Mat histogram, result;

        // Binary Image
        Mat binary_img;
        cv::threshold( img, binary_img, 128.0, 255.0, THRESH_BINARY );

        // Taks a histogram of the image
        cv::reduce( binary_img, histogram, 0, REDUCE_SUM, CV_32F );

        cvtColor( histogram, result, COLOR_GRAY2BGR );
        result = result * 255;

        // Find peaks of left and right halves of the histogram
        int midpoint = histogram.size().width / 2;

        int left_base = std::distance(
            histogram.begin<int>(),
            std::max_element( histogram.begin<int>(), histogram.begin<int>() + midpoint ) );
        int right_base = std::distance(
            histogram.begin<int>(),
            std::max_element( histogram.begin<int>() + midpoint, histogram.end<int>() ) );

        // Set the height of windows
        int window_height = img.rows / number_of_windows;
        int height = img.rows;
        // Find the x and y positions of all nonzero pixels in the image
        std::vector<cv::Point> nonzero_points;
        cv::findNonZero( binary_img, nonzero_points );

        // Current position to update the window
        int leftx_current = left_base;
        int rightx_current = right_base;

        int margin = 50;  // width of the windows + / -margin
        int minpix = 40;  // minimum number of pixels found to recenter window

        vector<Point> left_lane_points, right_lane_points;

        // Process through the window
        for( int window = 0; window < number_of_windows; window++ )
        {
            // Identify window boundaries in x and y( and right and left )
            int win_y_low = height - ( window + 1 ) * window_height;
            int win_y_high = height - window * window_height;
            int win_xleft_low = leftx_current - margin;
            int win_xleft_high = leftx_current + margin;
            int win_xright_low = rightx_current - margin;
            int win_xright_high = rightx_current + margin;

            // draw the left window
            cv::rectangle( binary_img, Point( win_xleft_low, win_y_low ),
                           Point( win_xleft_high, win_y_high ), Scalar( 255, 0, 0 ), 2 );

            // draw the right window
            cv::rectangle( binary_img, Point( win_xright_low, win_y_low ),
                           Point( win_xright_high, win_y_high ), Scalar( 255, 0, 0 ), 2 );
            

            std::vector<int> good_left_inds;
            std::vector<int> good_right_inds;

            // Step through window one by one
            for( int i = 0; i < nonzero_points.size(); ++i )
            {
                // filter the point whose ylabel satify the window
                if( nonzero_points[i].y >= win_y_low && nonzero_points[i].y < win_y_high )
                {
                    // filter the left points
                    if( nonzero_points[i].x >= win_xleft_low &&
                        nonzero_points[i].x < win_xleft_high )
                    {
                        good_left_inds.push_back( i );
                        left_lane_points.push_back( nonzero_points[i] );
                    }
                    // filter the right points
                    if( nonzero_points[i].x >= win_xright_low &&
                        nonzero_points[i].x < win_xright_high )
                    {
                        good_right_inds.push_back( i );
                        right_lane_points.push_back( nonzero_points[i] );
                    }
                }
            }

            // lane_ins records all satisfied point
            vector<vector<int>> left_lanes_inds, right_lanes_inds;
            left_lanes_inds.push_back( good_left_inds );
            right_lanes_inds.push_back( good_right_inds );

            // If you found > minpix pixels, recenter next window on their mean position
            if( good_left_inds.size() > minpix )
            {
                int sum = 0;
                for( int i = 0; i < good_left_inds.size(); i++ )
                {
                    sum += nonzero_points[good_left_inds[i]].x;
                }
                leftx_current = sum / good_left_inds.size();
            }

            if( good_right_inds.size() > minpix )
            {
                int sum = 0;
                for( int i = 0; i < good_right_inds.size(); i++ )
                {
                    sum += nonzero_points[good_right_inds[i]].x;
                }
                rightx_current = sum / good_right_inds.size();
            }
        }
       
         vector<Point> fitted_points = polyfit(left_lane_points,2);
       
  
        polylines( original_image_,fitted_points, false, Scalar( 255, 0, 0 ), 10 );
        imshow( "BEV", original_image_ );
       
    }

    vector<Point> polyfit( std::vector<cv::Point>& points, int degree )
    {
        if( points.empty() )
            throw std::invalid_argument( "Points vector is empty" );

        int numCoefficients = degree + 1;
        size_t nCount = points.size();

        cv::Mat X( nCount, numCoefficients, CV_64F );
        cv::Mat Y( nCount, 1, CV_64F );

        for( size_t i = 0; i < nCount; i++ )
        {
            Y.at<double>( i, 0 ) = points[i].y;

            double val = 1;
            for( int j = 0; j < numCoefficients; j++ )
            {
                X.at<double>( i, j ) = val;
                val *= points[i].x;
            }
        }

        cv::Mat Xt, XtX, XtY;
        transpose( X, Xt );
        XtX = Xt * X;
        XtY = Xt * Y;

        cv::Mat coefficients;
        cv::solve( XtX, XtY, coefficients, cv::DECOMP_LU );

        std::vector<double> result;
        result.reserve( numCoefficients );
        for( int i = 0; i < numCoefficients; i++ )
        {
            result.push_back( coefficients.at<double>( i, 0 ) );
        }

        // Create two vectors to store x,y
        vector<int> points_x, points_y;
        for( int i = 0; i < points.size(); i++ )
        {
            points_x.push_back( points[i].x );
            points_y.push_back( points[i].y );
        }

        // Find min, max value in x

        auto minIt = min_element( points_x.begin(), points_x.end() );
        auto maxIt = max_element( points_x.begin(), points_x.end() );

        int min_x = *minIt;
        int max_x = *maxIt;

        // Calculate the fitted y value
        vector<Point> fitted_line_points;
        for( int i = min_x; i < max_x; i += 1 )
        {
            Point point;
            point.x = i;
            point.y = result[0] + result[1] * point.x + result[2] * point.x * point.x;
            fitted_line_points.push_back( point );
        }

        return fitted_line_points;
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
                case tronis::TronisDataType::Object:
                {
                    processObject();
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
        // showImage( image_name_, detectLanes(image_) );
        showImage( image_name_, image_ );
        showImage( "Bird Eye View", BEV_image_ );
    }
};

//// main loop opens socket and listens for incoming data
// int main( int argc, char** argv )
//{
//    std::cout << "Welcome to lane assistant" << std::endl;
//
//    // specify socket parameters
//    std::string socket_type = "TcpSocket";
//    std::string socket_ip = "127.0.0.1";
//    std::string socket_port = "50542";
//
//    std::ostringstream socket_params;
//    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip
//                  << "\", PortBind:" << socket_port << "}";
//
//    int key_press = 0;  // close app on key press 'q'
//    tronis::CircularMultiQueuedSocket msg_grabber;
//    uint32_t timeout_ms = 500;  // close grabber, if last received msg is older than this param
//
//    LaneAssistant lane_assistant;
//
//    while( key_press != 'q' )
//    {
//        std::cout << "Wait for connection..." << std::endl;
//        msg_grabber.open_str( socket_params.str() );
//
//        if( !msg_grabber.isOpen() )
//        {
//            printf( "Failed to open grabber, retry...!\n" );
//            continue;
//        }
//
//        std::cout << "Start grabbing" << std::endl;
//        tronis::SocketData received_data;
//        uint32_t time_ms = 0;
//
//        while( key_press != 'q' )
//        {
//            // wait for data, close after timeout_ms without new data
//            if( msg_grabber.tryPop( received_data, true ) )
//            {
//                // data received! reset timer
//                time_ms = 0;
//
//                // convert socket data to tronis model data
//                tronis::SocketDataStream data_stream( received_data );
//                tronis::ModelDataWrapper data_model(
//                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
//                if( !data_model.is_valid() )
//                {
//                    std::cout << "received invalid data, continue..." << std::endl;
//                    continue;
//                }
//                // identify data type
//                lane_assistant.getData( data_model );
//                lane_assistant.processData( msg_grabber );
//            }
//            else
//            {
//                // no data received, update timer
//                ++time_ms;
//                if( time_ms > timeout_ms )
//                {
//                    std::cout << "Timeout, no data" << std::endl;
//                    msg_grabber.close();
//                    break;
//                }
//                else
//                {
//                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
//                    key_press = cv::waitKey( 1 );
//                }
//            }
//        }
//        msg_grabber.close();
//    }
//    return 0;
//}

int main( int argc, char** argv )
{
    Mat original_img = imread( "C:\\Users\\am3s33\\Pictures\\Camera Roll\\lane2.png" );
    LaneAssistant laneAssistant;
    laneAssistant.processImage( original_img );
}