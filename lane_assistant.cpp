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
        cv::Mat processd_img;
        processd_img = regionDetection( image );
        processd_img = colorDetection( processd_img );
        processd_img = edgeDetection( processd_img );
        processd_img = lineDetection( processd_img );
        imshow( "lane detection", processd_img );
        waitKey( 0 );
    }

protected:
    std::string image_name_;
    cv::Mat image_;
    tronis::LocationSub ego_location_;
    tronis::OrientationSub ego_orientation_;
    double ego_velocity_;

    // Function to detect lanes based on camera image
    // Insert your algorithm here
    void detectLanes()
    {
        // edgeDetection( image_ );
        // lineDetection();
        // do stuff
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

    void colorThreshold( const cv::Mat& src, Mat& dst, const Scalar& lower, const Scalar& upper )
    {
        Mat hsv;
        cv::cvtColor( src, hsv, COLOR_BGR2HSV );
        cv::inRange( hsv, lower, upper, dst );
    }

    Mat regionDetection( Mat img )
    {
        Mat result, mask_three_channel;
        int img_height = img.size().height;
        int img_width = img.size().width;
        /// mask image with polygon
        Mat mask_crop( img_height, img_width, CV_8UC1, Scalar( 0 ) );
        /// create cone shape
        vector<Point> pts;
        vector<vector<Point>> v_pts;
        pts.push_back( Point( 0, img_height / 2 + 90 ) );
        pts.push_back( Point( img_width / 2, img_height / 2 + 45 ) );
        pts.push_back( Point( img_width, img_height / 2 + 90 ) );
        pts.push_back( Point( img_width, img_height - 60 ) );
        pts.push_back( Point( img_width * 0.7, img_height - 125 ) );
        pts.push_back( Point( img_width * 0.3, img_height - 125 ) );
        pts.push_back( Point( 0, img_height - 60 ) );
        v_pts.push_back( pts );
        /// add cone to mask
        fillPoly( mask_crop, v_pts, 255 );

        cvtColor( mask_crop, mask_three_channel,
                  COLOR_GRAY2BGR );  // transform mask into three channels

        cv::bitwise_and( img, mask_three_channel, result );
        // image_orig.copyTo( temp, mask_crop2 ); //if color channel and mask

        return result;
    }
    Mat colorDetection( cv::Mat& img )
    {
        Mat mask_yellow, mask_white, mask_combined;

        // yellow region hsv threshold
        cv::Scalar lower_yellow( 20, 100, 100 );
        cv::Scalar upper_yellow( 30, 255, 255 );
        colorThreshold( img, mask_yellow, lower_yellow, upper_yellow );

        // white region hsv threshold
        cv::Scalar lower_white( 0, 10, 50 );
        cv::Scalar upper_white( 180, 70, 255 );
        colorThreshold( img, mask_white, lower_white, upper_white );

        cv::bitwise_or( mask_yellow, mask_white, mask_combined );
        return mask_combined;
    }
    cv::Mat edgeDetection( cv::Mat image )
    {
        cv::Mat grad_img, blurred_img;
        /* Solber function
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
cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

*/
        cv::GaussianBlur( image, blurred_img, cv::Size( 3, 3 ), 0, 0, cv::BORDER_DEFAULT );
        cv::Canny( image, grad_img, 50, 200 );
        return grad_img;
    }

    cv::Mat lineDetection( cv::Mat img )
    {
        Mat img_color;
        std::vector<Vec4i> lines;
        cv::HoughLinesP( img, lines, 6, CV_PI / 60, 200, 40, 20 );

        cvtColor( img, img_color, COLOR_GRAY2BGR );

        // painting lines in original image
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            line( img_color, Point( l[0], l[1] ), Point( l[2], l[3] ), Scalar( 0, 0, 255 ), 3,
                  LINE_AA );
        }

        return img_color;
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
        if( image.empty() )
        {
            std::cout << "empty image" << std::endl;
            return false;
        }

        image_name_ = base_name;
        image_ = tronis::image2Mat( image );

        detectLanes();
        // showImage( image_name_, detectLanes(image_) );
        showImage( image_name_, image_ );
    }
};
/*
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
*/
int main( int argc, char** argv )
{
    Mat original_img = imread( "C:\\Users\\am3s33\\Pictures\\Camera Roll\\lane1.png" );
    LaneAssistant laneAssistant;
    laneAssistant.processImage( original_img );
}