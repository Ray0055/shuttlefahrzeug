#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>

using namespace std;

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
                    std::cout << "Processing data..." << std::endl;
			// send results via socket
			return true;
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
			// do stuff
        }
		
        bool processPoseVelocity( tronis::PoseVelocitySub* msg)
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

// Helper functions, no changes needed
    public:
		// Function to process received tronis data
		bool getData( tronis::ModelDataWrapper data_model )
		{
            if( data_model->GetModelType() == tronis::ModelType::Tronis )
            {
                std::cout << "Id: " << data_model->GetTypeId()
                            << ", Name: " << data_model->GetName()
                            << ", Time: " << data_model->GetTime() << std::endl;

                // if data is sensor output, process data
                switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
                {
                    case tronis::TronisDataType::Image:
                    {
                        processImage(
                            data_model->GetName(),
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
            std::cout << "processImage" << std::endl;
            if( image.empty() )
            {
                std::cout << "empty image" << std::endl;
                return false;
            }

            image_name_ = base_name;
            image_ = tronis::image2Mat( image );

            detectLanes();
            showImage( image_name_, image_ );

            return true;
        }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    std::cout << "Welcome to lane assistant" << std::endl;

	// specify socket parameters
	std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
        std::string socket_port = "50542" ;

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip << "\", PortBind:" << socket_port << "}";

    int key_press = 0;	// close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500; // close grabber, if last received msg is older than this param

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
