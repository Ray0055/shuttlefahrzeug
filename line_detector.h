#define LINE_DETECTOR_h
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace std;
using namespace cv;

class Lane
{
public:
    Lane()
    {
    }
    // flag to mark if the line was detected the last iteration
    bool detected = false;

    // polynomial coefficients fitted on the last iteration
    Mat last_fit_pixel;  // coef in pixel
    Mat last_fit_meter;  // coef in real world

    // detected pixels on the last iteration
    vector<Point> last_lane_points_pixel;
    vector<Point> last_lane_points_meter;

	// center pixels on the last iteration
    vector<Point> last_center_pts_pixel;
    vector<Point> last_center_pts_meter;
};

void line_dection_by_sliding_window( Mat img, Mat output_img, int number_of_windows,
                                     Lane& left_lane, Lane& right_lane );

// polynomial line fit
Mat PolynomialFit( vector<Point>& points, int order );

Mat birdEyeView_fullscreen( Mat img, bool mode );

// compute fitted polynomial line points
vector<Point> computeFittedLinePoints( vector<Point>& points, Mat coefficient, int height );

// draw histogram as a helper function
void draw_histogram( Mat histogram );

// detect lanes by previous detected lanes to speed up the search of lane-lines
void line_detection_by_previous( Mat BEV_img, Mat output_img, int number_of_windows,
                                 Lane& left_lane, Lane& right_lane );

void draw_detected_lane_onto_road( Mat original_img, Lane& left_lane, Lane& right_lane );

// draw center lane onto road
void get_center_of_road( Mat detected_lane_img_meter, Mat detected_lane_img_BEV, Lane& left_lane,
                         Lane& right_lane );
