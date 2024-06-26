#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>

#include "line_detector.h"
using namespace std;
using namespace cv;

void line_dection_by_sliding_window( Mat img, Mat output_img, int number_of_windows,
                                     Lane& left_lane, Lane& right_lane )
{
    Mat histogram, result;

    // Binary Image
    Mat binary_img;
    cv::threshold( img, binary_img, 128.0, 255.0, THRESH_BINARY );

    // Taks a histogram of the image
    cv::reduce( binary_img, histogram, 0, REDUCE_SUM, CV_32F );

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
    int height = img.rows + 50;

    // Find the x and y positions of all nonzero pixels in the image
    std::vector<cv::Point> nonzero_points;
    cv::findNonZero( binary_img, nonzero_points );

    // Current position to update the window
    int leftx_current = left_base;
    int rightx_current = right_base;

    int margin = 20;   // width of the windows + / -margin
    int minpix = 300;  // minimum number of pixels found to recenter window

    vector<Point> left_lane_points, right_lane_points;

    // Process through the window
    for( int window = 0; window < number_of_windows; window++ )
    {
        // Identify window boundaries in x and y( and right and left )
        int win_y_low = ( window + 1 ) * window_height;
        int win_y_high = window * window_height;

        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;

        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;

        // draw the left window
        //cv::rectangle( img, Point( win_xleft_low, win_y_low ), Point( win_xleft_high, win_y_high ),
        //               Scalar( 255, 0, 0 ), 2 );

        //// draw the right window
        //cv::rectangle( img, Point( win_xright_low, win_y_low ),
        //               Point( win_xright_high, win_y_high ), Scalar( 255, 0, 0 ), 2 );

        std::vector<int> good_left_inds;
        std::vector<int> good_right_inds;

        // Step through window one by one
        for( int i = 0; i < nonzero_points.size(); ++i )
        {
            // filter the point whose ylabel satify the window
            if( nonzero_points[i].y <= win_y_low && nonzero_points[i].y > win_y_high )
            {
                // filter the left points
                if( nonzero_points[i].x >= win_xleft_low && nonzero_points[i].x < win_xleft_high )
                {
                    good_left_inds.push_back( i );
                    left_lane_points.push_back( nonzero_points[i] );
                }
                // filter the right points
                if( nonzero_points[i].x >= win_xright_low && nonzero_points[i].x < win_xright_high )
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
            if(leftx_current < midpoint)
            {
                leftx_current = sum / good_left_inds.size();
            }
            
        }

        if( good_right_inds.size() > minpix )
        {
            int sum = 0;
            for( int i = 0; i < good_right_inds.size(); i++ )
            {
                sum += nonzero_points[good_right_inds[i]].x;
            }
            if( rightx_current > midpoint )
            {
                rightx_current = sum / good_right_inds.size();
            }
             
        }
    }
    cout << "Now in polyfit.." << endl;

    // polynomial line fit
    if( right_lane_points.size() > minpix )
    {
        Mat right_fit_coeff = PolynomialFit( right_lane_points, 2 );

        right_lane_points.push_back( Point( 0, 0 ) );
        vector<Point> fitted_points =
            computeFittedLinePoints( right_lane_points, right_fit_coeff, height );

        // Update lane data to class Lane
        right_lane.last_fit_pixel = right_fit_coeff;
        right_lane.last_lane_points_pixel = fitted_points;
        right_lane.detected = true;

       // cv::polylines( output_img, fitted_points, false, Scalar( 255, 0, 0 ), 10 );

        cout << "right lines are detected by sliding windows, the number of pixel is"
             << right_lane_points.size() << endl;
    }
    else
    {
        right_lane.detected = false;
        cout << "right lines are not detected." << endl;
    }

    if( left_lane_points.size() > minpix )
    {
        Mat left_fit_coeff = PolynomialFit( left_lane_points, 2 );

        //// Check if the coeff correctly
        //      if( left_fit_coeff.at<double>( 0, 0 ) < 195 || left_fit_coeff.at<double>( 0, 0 ) >
        //      350 )
        //      {
        //          left_fit_coeff = left_lane.last_fit_pixel;
        //      }

        // Compute fitted Points
        left_lane_points.push_back( Point( 0, 0 ) );
        vector<Point> fitted_points =
            computeFittedLinePoints( left_lane_points, left_fit_coeff, height );

        // Update lane data to class Lane
        left_lane.last_fit_pixel = left_fit_coeff;
        left_lane.last_lane_points_pixel = fitted_points;
        left_lane.detected = true;

        //cv::polylines( output_img, fitted_points, false, Scalar( 0, 255, 0 ), 10 );
        cout << "left lines are detected by sliding windows, the number of pixel is"
             << left_lane_points.size() << endl;
    }

    else
    {
        left_lane.detected = false;
        cout << "left lines are not detected." << endl;
    }

    // draw_histogram( histogram );
}

void line_detection_by_previous( Mat BEV_img, Mat output_img, int number_of_windows,
                                 Lane& left_lane, Lane& right_lane )
{
    std::cout << "Now in line detecting by previous" << endl;
    Mat histogram, result;
    int height = output_img.cols + 50;
    // Binary Image
    Mat binary_img;
    cv::threshold( BEV_img, binary_img, 128.0, 255.0, THRESH_BINARY );

    // get fitting coefficient from previous data
    Mat left_fit_coeff = left_lane.last_fit_pixel;
    Mat right_fit_coeff = right_lane.last_fit_pixel;

    // Find the x and y positions of all nonzero pixels in the image
    std::vector<cv::Point> nonzero_points;
    cv::findNonZero( binary_img, nonzero_points );

    int margin = 20;
    int minpix = 300;  // minimum number of pixels found to recenter window

    vector<int> left_lane_points_index, right_lane_points_index;
    vector<Point> left_lane_points, right_lane_points;

    /*
Using the polynomial calculated from the previous frame to create a search area, and determining
the position of the lane lines by comparing whether the points detected in the current frame are
within this search area, this method speeds up the search and enhances stability.
    */

    for( int i = 0; i < nonzero_points.size(); i++ )
    {
        int px = nonzero_points[i].x;
        int py = nonzero_points[i].y;

        // point.x computed by prevous fitting coefficient
        double px_pre_left = left_fit_coeff.at<double>( 0, 0 ) +
                             left_fit_coeff.at<double>( 1, 0 ) * py +
                             left_fit_coeff.at<double>( 2, 0 ) * py * py;
        double px_pre_right = right_fit_coeff.at<double>( 0, 0 ) +
                              right_fit_coeff.at<double>( 1, 0 ) * py +
                              right_fit_coeff.at<double>( 2, 0 ) * py * py;

        if( px > px_pre_left - margin && px < px_pre_left + margin )
        {
            left_lane_points_index.push_back( i );
            left_lane_points.push_back( nonzero_points[i] );
            // cout << "left lane have" << left_lane_points.size() << endl;
        }

        if( px > px_pre_right - margin && px < px_pre_right + margin )
        {
            right_lane_points_index.push_back( i );
            right_lane_points.push_back( nonzero_points[i] );
            // cout << "right lane have" << right_lane_points.size() << endl;
        }
    }

    // Determining whether the lane is detected
    // If lane is not detected or too less partly detected
    if( left_lane_points_index.size() < minpix )
    {
        left_lane.detected = false;
        cout << "left lane was not detected, minpix is:" << left_lane_points.size() << endl;
    }

    //if lane is not correctly detected
       else if( left_fit_coeff.at<double>( 0, 0 ) > 270 
       )
       {
          
           left_lane.detected = false;
          
       }
    else
    {
        // compute new fitting coefficient
        left_fit_coeff = PolynomialFit( left_lane_points, 2 );
        cout << "coeff is: " << left_fit_coeff.at<double>( 0, 0 ) << ","
             << left_fit_coeff.at<double>( 1, 0 ) << "," << left_fit_coeff.at<double>( 2, 0 )
             << endl;
        // update to the lane
        left_lane.last_fit_pixel = left_fit_coeff;
        left_lane.detected = true;

        // draw polyline to the image
        left_lane_points.push_back( Point( 0, 0 ) );
        vector<Point> fitted_points =
            computeFittedLinePoints( left_lane_points, left_fit_coeff, height );
        //cv::polylines( output_img, fitted_points, false, Scalar( 0, 255, 0 ), 10 );
        left_lane.last_lane_points_pixel = fitted_points;

        cout << "Left lane is detected by previous detection. " << endl;
    }

    // if right lane was not detected or detected pixel are too less
    if( right_lane_points_index.size() < minpix )
    {
        right_lane.detected = false;
        cout << "right lane was not detected, minpix is:" << right_lane_points.size() << endl;
    }
    // if lane is not correctly detected
     else if( right_fit_coeff.at<double>( 0, 0 ) < 450 || right_fit_coeff.at<double>( 0, 0 ) > 600
     )
    {
        right_lane.detected = false;       
    }
    else
    {
        // compute new fitting coefficient
        right_fit_coeff = PolynomialFit( right_lane_points, 2 );
        cout << "coeff is: " << right_fit_coeff.at<double>( 0, 0 ) << ","
             << right_fit_coeff.at<double>( 1, 0 ) << "," << right_fit_coeff.at<double>( 2, 0 )
             << endl;
        // update to the lane
        right_lane.last_fit_pixel = right_fit_coeff;
        right_lane.detected = true;

        // draw polyline to the image
        right_lane_points.push_back( Point( 0, 0 ) );
        vector<Point> fitted_points =
            computeFittedLinePoints( right_lane_points, right_fit_coeff, height );
        //cv::polylines( output_img, fitted_points, false, Scalar( 255, 0, 0 ), 10 );
        right_lane.last_lane_points_pixel = fitted_points;

        cout << "right lane is detected by previous detection. " << endl;
    }
}
Mat PolynomialFit( vector<Point>& points, int order )
{
    Mat U( points.size(), ( order + 1 ), CV_64F );
    Mat Y( points.size(), 1, CV_64F );
    for( int i = 0; i < U.rows; i++ )
    {
        for( int j = 0; j < U.cols; j++ )
        {
            U.at<double>( i, j ) = pow( points[i].y, j );
        }
    }

    for( int i = 0; i < Y.rows; i++ )
    {
        Y.at<double>( i, 0 ) = points[i].x;
    }

    Mat K( ( order + 1 ), 1, CV_64F );
    if( U.data != NULL )
    {
        K = ( U.t() * U ).inv() * U.t() * Y;
    }

    return K;
}

vector<Point> computeFittedLinePoints( vector<Point>& points, Mat coefficient, int height )
{
    vector<Point> fitted_line_points;
    // Create two vectors to store x,y
    vector<int> points_x, points_y;
    for( int i = 0; i < points.size(); i++ )
    {
        points_x.push_back( points[i].x );
        points_y.push_back( points[i].y );
    }

    // Find min, max value in x
    auto minIt = min_element( points_y.begin(), points_y.end() );
    auto maxIt = max_element( points_y.begin(), points_y.end() );

    int min_y = *minIt;
    int max_y = *maxIt;

    // Calculate the fitted y value

    for( int i = min_y; i < height; i++ )
    {
        Point point;
        point.y = i;
        point.x = coefficient.at<double>( 0, 0 ) + coefficient.at<double>( 1, 0 ) * point.y +
                  coefficient.at<double>( 2, 0 ) * point.y * point.y;
        fitted_line_points.push_back( point );
    }

    return fitted_line_points;
}

Mat birdEyeView_fullscreen( Mat img, bool mode )
{
    /*Create bird eye view*/

    int image_height = img.size().height;
    int image_width = img.size().width;

    vector<Point2f> src( 4 ), dst( 4 );
    src[0] = Point2f( image_width / 2 - 130, image_height / 2 + 45 );  // top left
    src[1] = Point2f( 0, image_height - 100 );                         // bottom left
    src[2] = Point2f( image_width, image_height - 100 );               // bottom right
    src[3] = Point2f( image_width / 2 + 130, image_height / 2 + 45 );  // top right

    dst[0] = Point2f( image_width / 2 - 180, 0 );             // top left
    dst[1] = Point2f( image_width / 2 - 180, image_height );  // bottom left
    dst[2] = Point2f( image_width / 2 + 180, image_height );  // bottom right
    dst[3] = Point2f( image_width / 2 + 180, 0 );             // top right

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

void draw_detected_lane_onto_road( Mat original_img, Lane& left_lane, Lane& right_lane )
{
    int image_height = original_img.size().height;
    int image_width = original_img.size().width;

    vector<Point2f> src( 4 ), dst( 4 );
    src[0] = Point2f( image_width / 2 - 130, image_height / 2 + 45 );  // top left
    src[1] = Point2f( 0, image_height - 100 );                         // bottom left
    src[2] = Point2f( image_width, image_height - 100 );               // bottom right
    src[3] = Point2f( image_width / 2 + 130, image_height / 2 + 45 );  // top right

	dst[0] = Point2f( image_width / 2 - 180, 0 );             // top left
    dst[1] = Point2f( image_width / 2 - 180, image_height );  // bottom left
    dst[2] = Point2f( image_width / 2 + 180, image_height );  // bottom right
    dst[3] = Point2f( image_width / 2 + 180, 0 );             // top right

    Mat M = getPerspectiveTransform( src, dst );
    Mat Minv = getPerspectiveTransform( dst, src );

    vector<Point> left_pts_pixel = left_lane.last_lane_points_pixel;
    vector<Point> right_pts_pixel = right_lane.last_lane_points_pixel;
    vector<Point> center_pts_pixel;

    /*if( left_pts_pixel.size() == right_pts_pixel.size() )
    {
        for( int i = 0; i < left_pts_pixel.size(); i++ )
        {
            int px = ( left_pts_pixel.at( i ).x + right_pts_pixel.at( i ).x ) / 2;
            int py = left_pts_pixel.at( i ).y;
            center_pts_pixel.push_back( Point( px, py ) );
        }
    }*/
    vector<Point> left_pts_meter =
        applyPerspectiveTransform( left_lane.last_lane_points_pixel, Minv );
    vector<Point> right_pts_meter =
        applyPerspectiveTransform( right_lane.last_lane_points_pixel, Minv );
    /*vector<Point> center_pts_meter = applyPerspectiveTransform( center_pts_pixel, Minv );*/

    // update lane points in meter
    left_lane.last_lane_points_meter = left_pts_meter;
    right_lane.last_lane_points_meter = right_pts_meter;
    /*right_lane.last_center_pts_pixel = center_pts_pixel;*/

    //// change the Reihefolge der Points
    // std::reverse( inv_right_pts.begin(), inv_right_pts.end() );

    //// create bottom line points
    // vector<Point> bottom_point;
    // for( int i = 0; i < image_width; i++ )
    //{
    //    bottom_point.push_back( Point( i, image_height ) );
    //}

    //// create right edge points
    // vector<Point> right_edge_pts;
    // for( int i = inv_right_pts.front().y; i < image_height; i++ )
    //{
    //    right_edge_pts.push_back( Point( image_width, i ) );
    //}

    //// create left edge points
    // vector<Point> left_edge_pts;
    // for( int i = inv_left_pts.back().y; i < image_height; i++ )
    //{
    //    left_edge_pts.push_back( Point( 0, i ) );
    //}

    //// create top line points
    // vector<Point> top_edge_pts;
    // Point p1 = inv_right_pts.back();
    // Point p2 = inv_left_pts.front();
    // int dx = abs( p2.x - p1.x ), sx = p1.x < p2.x ? 1 : -1;
    // int dy = -abs( p2.y - p1.y ), sy = p1.y < p2.y ? 1 : -1;
    // int err = dx + dy, e2;

    // while( true )
    //{
    //    top_edge_pts.push_back( p1 );
    //    if( p1.x == p2.x && p1.y == p2.y )
    //        break;
    //    e2 = 2 * err;
    //    if( e2 >= dy )
    //    {
    //        err += dy;
    //        p1.x += sx;
    //    }
    //    if( e2 <= dx )
    //    {
    //        err += dx;
    //        p1.y += sy;
    //    }
    //}

    //// merge all lines
    // vector<vector<Point>> polygon;
    // vector<Point> pts;
    // pts.insert( pts.begin(), inv_left_pts.begin(), inv_left_pts.end() );
    // pts.insert( pts.end(), left_edge_pts.begin(), left_edge_pts.end() );
    // pts.insert( pts.end(), bottom_point.begin(), bottom_point.end() );
    // pts.insert( pts.end(), right_edge_pts.begin(), right_edge_pts.end() );
    // pts.insert( pts.end(), inv_right_pts.begin(), inv_right_pts.end() );
    // pts.insert( pts.end(), top_edge_pts.begin(), top_edge_pts.end() );

    // polygon.push_back( pts );

    // Mat mask_fill( original_img.size(), original_img.type() );
    // fillPoly( mask_fill, polygon, Scalar( 127, 236, 155 ) );

    // cv::addWeighted( original_img, 1, mask_fill, 0.3, 0.0, original_img );

    // draw left, right lane to original image(real road)
    if( left_lane.detected && right_lane.detected )
    {

        cv::polylines( original_img, left_pts_meter, false, Scalar( 255, 0, 0 ), 8 );
        cv::polylines( original_img, right_pts_meter, false, Scalar( 0, 255, 0 ), 8 );
        cout << left_pts_meter.empty() << endl;
		cout << "draw both left and right lane" << endl;
        
    }
    else if( left_lane.detected && !right_lane.detected )
    {
        cv::polylines( original_img, left_pts_meter, false, Scalar( 255, 0, 0 ), 8 );
        cout << "draw only left lane." << endl;
    }
    else if( !left_lane.detected && right_lane.detected )
    {
        cv::polylines( original_img, right_pts_meter, false, Scalar( 0, 255, 0 ), 8 );
        cout << "draw only right lane." << endl;
    }

    // cv::polylines( original_img, center_pts_meter, false, Scalar( 246, 161, 75 ), 5 );

    imshow( "origial", original_img );
}

vector<Point> applyPerspectiveTransform( const vector<Point>& inputPoints,
                                         const Mat& transformationMatrix )
{
    // coonvert integer to float point
    vector<Point2f> inputPointsFloat;
    inputPointsFloat.reserve( inputPoints.size() );
    for( const Point& p : inputPoints )
    {
        inputPointsFloat.push_back(
            Point2f( static_cast<float>( p.x ), static_cast<float>( p.y ) ) );
    }

    // perspective transform apply
    vector<Point2f> transformedPointsFloat;
    cv::perspectiveTransform( inputPointsFloat, transformedPointsFloat, transformationMatrix );

    // convert float to integer
    vector<Point> transformedPoints;
    transformedPoints.reserve( transformedPointsFloat.size() );
    for( const Point2f& pt : transformedPointsFloat )
    {
        transformedPoints.push_back( Point( cvRound( pt.x ), cvRound( pt.y ) ) );
    }

    return transformedPoints;
}
