#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//pcl
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/filters/filter.h>

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

//#include "persistence1d.hpp"

#include <algorithm>

using namespace cv;
using namespace std;
//using namespace p1d;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::Normal> PointCloudNormal;

struct Point1D{
	Point1D(const int& index_ = 0, const int& var_ = 0) :index(index_), var(var_){}
	int index;
	int var;
};

struct Point1DCluster{
	Point1DCluster(const int& median_ = 0) : median(median_){}
	vector<Point1D> pts;
	int median;
};

class pc_seg
{

	friend void display(const pc_seg& seg, const string& window);

public:
	Mat depMap_;
	Mat depMap_U;
	Mat depMap_stretch;
	Mat depMap_morph;
	Mat depMap_HE; //increase contrast by histogram equalization
	Mat depMap_boundary;
	Mat depMap_binary;
	Mat depMap_DT;
	Mat depMap_gradie;
	PointCloudXYZ pts_;
	PointCloudNormal normal_;
public:
	pc_seg(const string PCD_file) :
		PCD_file_(PCD_file),
		is_init_(false)
	{
		if (read_PCD2Mat(PCD_file_, depMap_, pts_) == 0)
			is_init_ = true;
		else
		{
			is_init_ = false; PCL_ERROR("Initiate object failed!");
		}
	}

	void morphology_proc(const Mat& input, Mat& output, const int& morphType, const int& morphShape, const int& size, const int iterations)
	{
		Mat output_;
		if (is_init_)
		{
			Mat eleStruct = getStructuringElement(morphShape, Size(size, size));
			morphologyEx(input, output_, morphType, eleStruct, cv::Point(-1, -1), iterations);
			output_.copyTo(output);
		}
	}

	void find_boundary(Mat& input, Mat& output, const int& thresh)
	{
		Mat input_;
		if (input.depth() == CV_32F)
			input.convertTo(input_, CV_8U, 255);
		else
			input.copyTo(input_);
		Canny(input_, output, thresh, thresh * 2, 3, true);
	}

	void histoEqual_proc(const Mat& input, Mat& output)
	{
		Mat input_;
		if (input.depth() == CV_32F)
			input.convertTo(input_, CV_8U, 255);
		else
			input.copyTo(input_);
		equalizeHist(input_, output);
	}

	void contrast_stretching(const Mat& input, Mat& output, int lowOut, int highOut)
	{
		Mat input_;
		if (input.depth() == CV_32F)
			input.convertTo(input_, CV_8U, 255);
		else
			input.copyTo(input_);
		//CV_Assert(input.type()==CV_8UC1);
		//calculate histogram
		Mat hist;
		int histSize = 256;
		float range[] = { 0, 256 };
		const float* histRange = { range };
		calcHist(&input_, 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);

		vector<Point1D> pts;
		for (int i = 0; i < hist.rows; ++i)
		{
			int cnt = hist.at<float>(i, 0);
			if (cnt>0)
			{
				pts.push_back(Point1D(i, cnt));
			}
		}

		vector<Point1DCluster> clusters;
		Point1DCluster tmp_single_cluster;

		for (vector<Point1D>::iterator it = pts.begin(); it != pts.end(); it++){
			if (tmp_single_cluster.pts.empty()){
				tmp_single_cluster.pts.push_back(*it);
				//if this is the last element, then create a cluster and break
				if (it == pts.end() - 1){
					clusters.push_back(tmp_single_cluster);
					break;
				}
			}


			if (((it + 1)->index) - (it->index) <= 1)
			{
				tmp_single_cluster.pts.push_back(*(it + 1));
				//obvious that the last element has been processed, just quit the for loop
				if (it + 1 == pts.end() - 1){
					clusters.push_back(tmp_single_cluster);
					break;
				}

			}
			else
			{
				clusters.push_back(tmp_single_cluster);
				tmp_single_cluster.pts.clear();
			}
		}

		//not necessary for now
		//for (auto cluster : clusters){
		//	std::sort(cluster.pts.begin(), cluster.pts.end(),sortPoint1DCluster);
		//}
		//for (auto cluster : clusters)
		//{
		//	int mid_index = cluster.pts.size() / 2;
		//	cluster.median = cluster.pts[mid_index].index;
		//}

		clusters.erase(clusters.begin());
		clusters.erase(clusters.end() - 1);

		vector<Point1D> filtered_pts;


		//        for (auto single_cluster : clusters)
		//        {
		//            for (auto point : single_cluster.pts)
		//            {
		//                filtered_pts.push_back(point);
		//            }
		//        }

		for (vector<Point1DCluster>::iterator it = clusters.begin(); it != clusters.end(); ++it)
		{
			for (vector<Point1D>::iterator itt = it->pts.begin(); itt != it->pts.end(); ++itt)
			{
				filtered_pts.push_back(*itt);
			}
		}

		std::sort(filtered_pts.begin(), filtered_pts.end(), sortPoint1DCluster);

		//vector<float> hist_array;
		//for (int i = 0; i<hist.rows; i++)
		//	hist_array.push_back(hist.at<float>(i, 0));
		//not necessary 
		//Persistence1D p;
		//vector<int> minIndex, maxIndex;
		//p.RunPersistence(hist_array);
		//vector<TPairedExtrema> extrema;
		//p.GetPairedExtrema(extrema);
		//p.GetExtremaIndices(minIndex, maxIndex);
		//std::sort(minIndex.begin(), minIndex.end());
		//std::sort(extrema.begin(), extrema.end(), sortPersistence);
		////define contrast stretch params: lowIn & highIn


		int lowIn = (filtered_pts.begin())->index; int highIn = (filtered_pts.end() - 1)->index;
		output.create(input_.rows, input_.cols, CV_8UC1);
		for (int i = 0; i<input_.rows; i++)
		for (int j = 0; j<input_.cols; j++){
			float stretch = computeContrastStretch(input_.at<uchar>(i, j), lowIn, highIn, lowOut, highOut);
			output.at<uchar>(i, j) = saturate_cast<uchar>(stretch);
		}
	}

	void watershed_segmentation(Mat& criterion, Mat& marker, vector<vector<cv::Point> > contours, Mat& dst, Mat& label)
	{
		Mat dep_3CH, marker_;
		marker.copyTo(marker_);
		cvtColor(criterion, dep_3CH, COLOR_GRAY2BGR);
		watershed(dep_3CH, marker_);

		Mat mark = Mat::zeros(marker_.size(), CV_8UC1);
		marker_.convertTo(mark, CV_8UC1);
		bitwise_not(mark, mark);

		// Generate random colors
		vector<Vec3b> colors;
		for (size_t i = 0; i < contours.size(); i++)
		{
			int b = theRNG().uniform(0, 255);
			int g = theRNG().uniform(0, 255);
			int r = theRNG().uniform(0, 255);
			colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
		}

		// Create the result image
		dst = Mat::zeros(marker_.size(), CV_8UC3);
		// Fill labeled objects with random colors
		for (int i = 0; i < marker_.rows; i++)
		{
			for (int j = 0; j < marker_.cols; j++)
			{
				int index = marker_.at<int>(i, j);
				if (index > 0 && index <= static_cast<int>(contours.size()))
					dst.at<Vec3b>(i, j) = colors[index - 1];
				else
					dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}

		//std::vector<int> indice;
		//for (int i = 0; i < marker_.rows; i++)
		//{
		//	for (int j = 0; j < marker_.cols; j++)
		//	{
		//		int label = marker_.at<int>(i, j);
		//		if (label > 0 && label <= static_cast<int>(contours.size()))
		//		{
		//			indice.push_back(i*marker_.cols + j);
		//		}
		//	}
		//}
		mark.copyTo(label);

	}

	void distanceTransform_proc(const Mat& input, Mat& distance_transform, Mat& markers, const int& num_iter)
	{
		if (num_iter <= 0)
		{
			PCL_ERROR("The number of iterations must be larger than 01");
			return;
		}

		Mat distanceTransfrom_, binary_;

		for (int i = 0; i < num_iter; i++)
		{
			Mat tmp_input;
			if (i + 1 == 1){	//if this is the first time to process the image
				input.copyTo(tmp_input);
				distanceTransform(tmp_input, distanceTransfrom_, CV_DIST_L2, 3);
				cv::normalize(distanceTransfrom_, distanceTransfrom_, 0, 255, NORM_MINMAX);
				distanceTransfrom_.convertTo(distanceTransfrom_, CV_8U);
				//binarize image to gain MARKERS
				threshold(distanceTransfrom_, binary_, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			}
			else{
				binary_.copyTo(tmp_input);
				distanceTransform(tmp_input, distanceTransfrom_, CV_DIST_L2, 3);
				cv::normalize(distanceTransfrom_, distanceTransfrom_, 0, 255, NORM_MINMAX);
				distanceTransfrom_.convertTo(distanceTransfrom_, CV_8U);
				//binarize image to gain MARKERS
				threshold(distanceTransfrom_, binary_, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			}
		}
		distanceTransfrom_.copyTo(distance_transform);
		binary_.copyTo(markers);
	}

	void drawConvexHull(vector<vector<cv::Point> > contours, Mat& img)
	{
		//		for (auto single_contour : contours)
		//		{
		//			if (single_contour.size() > 100)
		//			{
		//				vector<cv::Point> convex;
		//				convexHull(single_contour, convex);
		//				polylines(img, convex, true, Scalar(255, 255, 255));
		//			}
		//		}

		for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
		{
			if (it->size() > 100)
			{
				vector<cv::Point> convex;
				convexHull(*it, convex);
				polylines(img, convex, true, Scalar(255, 255, 255));
			}
		}
	}

	void findAndDrawContours(Mat& edge_img, Mat& contour_img)
	{
		vector<vector<cv::Point> > contours;
		findContours(edge_img, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		//		for (auto single_contour : contours){
		//			if (single_contour.size() > 50){
		//				polylines(contour_img, single_contour, true, Scalar::all(255));
		//			}
		//		}

		for (vector<vector<cv::Point> >::iterator it = contours.begin(); it != contours.end(); ++it)
		{
			if (it->size() > 100)
			{
				polylines(contour_img, *it, true, Scalar::all(255));
			}
		}


	}

	void extractPointsByIndices(pcl::PointIndices::Ptr indices, const PointCloudXYZ::Ptr ref_pts, PointCloudXYZ::Ptr extracted_pts, bool is_negative)
	{
		//init
		//indices = boost::make_shared<pcl::PointIndices>();
		//extracted_pts = boost::make_shared<PointCloudXYZ>();

		pcl::ExtractIndices<pcl::PointXYZ> tmp_extractor;
		tmp_extractor.setInputCloud(ref_pts);
		tmp_extractor.setNegative(is_negative);
		tmp_extractor.setIndices(indices);
		tmp_extractor.filter(*extracted_pts);
	}

	void extractPointsByIndices(pcl::PointIndices::Ptr indices, const PointCloudNormal::Ptr ref_pts, PointCloudNormal::Ptr extracted_pts)
	{
		pcl::ExtractIndices<pcl::Normal> tmp_extractor;
		tmp_extractor.setInputCloud(ref_pts);
		tmp_extractor.setNegative(false);
		tmp_extractor.setIndices(indices);
		tmp_extractor.filter(*extracted_pts);
	}

	void integralImgNormalEstimation()
	{
		pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
		ne.setMaxDepthChangeFactor(0.02f);
		ne.setNormalSmoothingSize(10.0f);
		ne.setInputCloud(pts_.makeShared());
		ne.compute(normal_);

	}

	void integralImgNormalEstimation(const pcl::PointIndices::Ptr indices, PointCloudNormal::Ptr normal)
	{
		pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
		ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
		ne.setMaxDepthChangeFactor(0.02f);
		ne.setNormalSmoothingSize(10.0f);
		ne.setInputCloud(pts_.makeShared());
		//ne.setIndices(indices);
		ne.compute(*normal);
	}

	void plane_segment_proc(PointCloudXYZ::Ptr input_pts, float distance_thresh, pcl::PointIndices::Ptr inliers)
	{
		//plane segmentation
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		// Create the segmentation object
		pcl::SACSegmentation<pcl::PointXYZ> plane_seg;
		// Optional
		plane_seg.setOptimizeCoefficients(true);
		// Mandatory
		plane_seg.setModelType(pcl::SACMODEL_PLANE);
		plane_seg.setMethodType(pcl::SAC_RANSAC);
		plane_seg.setDistanceThreshold(distance_thresh);
		plane_seg.setInputCloud(input_pts);
		plane_seg.segment(*inliers, *coefficients);
	}

	int read_PCD(const std::string path, pcl::PointCloud<pcl::PointXYZ> &pts)
	{
		int result = pcl::io::loadPCDFile(path, pts);
		if (result == -1)
			PCL_ERROR("Could not load PCD file!");
		return result;
	}

	int read_PCD2Mat(const string& path, Mat& img, PointCloudXYZ& pts)
	{
		if (read_PCD(path, pts) == -1)
		{
			PCL_ERROR("Could not load PCD file!");
			return (-1);
		}
		else
		{
			img.create(pts.height, pts.width, CV_32F);
			int width = pts.width;
			for (int i = 0; i<pts.height; i++)
			for (int j = 0; j<pts.width; j++)
				img.at<float>(i, j) = pts.points[i*width + j].z;
			img.convertTo(depMap_U, CV_8UC1, 255);
			return 0;
		}
	}

	void read_PCD_save_Mat(const string& PCD_path, const string& Mat_path)
	{
		//read PCD to Mat
		Mat img, img_UCHAR;
		PointCloudXYZ pts;
		read_PCD2Mat(PCD_path, img, pts);
		//convert the image from CV_32F to CV_8U
		img.convertTo(img_UCHAR, CV_8U, 255);
		imwrite(Mat_path, img_UCHAR);
	}

private:
	string PCD_file_;
	bool is_init_;

private:
	float computeContrastStretch(const int& input, int lowIn, int highIn, int lowOut, int highOut)
	{
		float result;
		//origin contrast stretching
		if (0 <= input && input<lowIn){
			result = lowOut / lowIn*input;
		}
		else if (lowIn <= input && input <= highIn){
			result = ((highOut - lowOut) / (highIn - lowIn))*(input - lowIn) + lowOut;
		}
		else if (input>highIn && input <= 255){
			result = 0.0;
		}

		//only strectch the target
		//        if(lowIn<=input && input<=highIn){
		//           result=((highOut-lowOut)/(highIn-lowIn))*(input-lowIn)+lowOut;
		//        }else{
		//            result=0.0;
		//        }

		return result;
	}

	//	static  bool sortPersistence(const TPairedExtrema& i, const TPairedExtrema& j)
	//	{
	//		return(i.Persistence>j.Persistence);
	//	}

	static bool sortPoint1DCluster(const Point1D& i, const Point1D& j)
	{
		return(i.index < j.index);
	}

	static bool sortPoint1DClusterByVar(const Point1D& i, const Point1D& j)
	{
		return(i.var > j.var);
	}

};

void display(const pc_seg& seg, const string& window)
{
	if (seg.is_init_)
	{
		imshow(window, seg.depMap_);
		waitKey(0);
	}
	else
		PCL_ERROR("Object not initiated!");
}

void display(const Mat& img)
{
	imshow("test", img);
	waitKey(0);
}




int main()
{
	//define images
	Mat depMap_;
	Mat depMap_U;
	Mat depMap_stretch;
	Mat depMap_morph;
	Mat depMap_HE; //increase contrast by histogram equalization
	Mat depMap_boundary;
	Mat depMap_binary;
	Mat depMap_DT;
	Mat depMap_gradie;


    pc_seg seg("/home/yake/catkin_ws/src/segmentation/pcd/1476275797_pc.pcd");
	//using contrast stretch to increase contrast
	seg.depMap_U.copyTo(depMap_U);

	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	seg.plane_segment_proc(seg.pts_.makeShared(), 0.03, inliers);
	
	//seg.contrast_stretching(depMap_U, depMap_stretch, 100, 250);
	depMap_U.copyTo(depMap_stretch);
	//for (auto& it : inliers->indices)
	//{
	//	int i = it / depMap_stretch.cols;
	//	int j = it - i*depMap_stretch.cols;
	//	depMap_stretch.at <unsigned char>(i,j)= 0;
	//}
	for (std::vector<int>::iterator it = inliers->indices.begin(); it != inliers->indices.end();++it)
	{
		int i = *it / depMap_stretch.cols;
		int j = *it - i*depMap_stretch.cols;
		depMap_stretch.at <unsigned char>(i, j) = 0;
	}

	////binarize image
	//threshold(depMap_stretch, depMap_binary, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	////using close operation to fill holes
	//seg.morphology_proc(depMap_binary, depMap_morph, MORPH_CLOSE, MORPH_RECT, 3,3);

    //seg.morphology_proc(depMap_stretch, depMap_stretch, MORPH_CLOSE, MORPH_RECT, 3, 1);

	//using distance transform to gain markers
    seg.distanceTransform_proc(depMap_stretch, depMap_DT,depMap_binary, 3);

	//find contours
	vector<vector<cv::Point> > contours;
	findContours(depMap_binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//draw markers
	Mat markers = Mat::zeros(depMap_binary.size(), CV_32SC1);
	for (size_t i = 0; i<contours.size(); i++)
		drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
	circle(markers, Point(5, 5), 3, Scalar(255, 255, 255), -1);


//------------------------------------------------------------------------------------------------//
	////using depth image as criterion
	//Mat dist_depMap_U, dist_depMap_stretch;
	//Mat depMap_U_crop = depMap_U(cv::Rect(63, 0, 625, 480)); Mat markers_crop = markers(cv::Rect(63, 0, 625, 480));
	//circle(markers_crop, Point(5, 5), 3, Scalar(255, 255, 255), -1);
	//seg.watershed_segmentation(depMap_U_crop, markers_crop, contours, dist_depMap_U);
	//seg.watershed_segmentation(depMap_stretch, markers, contours, dist_depMap_stretch);

	////using distance transform as critertion
	//Mat dist_distance_transform;
	//bitwise_not(depMap_DT, depMap_DT);
	//seg.watershed_segmentation(depMap_DT, markers, contours, dist_distance_transform);
//------------------------------------------------------------------------------------------------//
	//extract edges
	//const int thresh = 8;
	//Mat edge_depMap_U, edge_depMap_stretch;
	//Mat depMap_U_crop = depMap_U(cv::Rect(63, 0, 625, 480)); Mat markers_crop = markers(cv::Rect(63, 0, 625, 480));
	//Canny(depMap_U_crop, edge_depMap_U, thresh, 2 * thresh);
	//Canny(depMap_stretch, edge_depMap_stretch, thresh, 2 * thresh);

	//find contours for 2 edge image
	vector<vector<cv::Point> > contourPoint_depMap_U, contourPoint_depMap_stretch;
	//Mat contour_depMap_U = Mat::zeros(depMap_U_crop.size(), CV_8UC1); 
	Mat contour_depMap_stretch = Mat::zeros(depMap_stretch.size(), CV_8UC1);
	//seg.findAndDrawContours(edge_depMap_U, contour_depMap_U);
	seg.findAndDrawContours(depMap_stretch, contour_depMap_stretch);

	//using contour image as criterion
	Mat contour_depMap_U_dilate, dist_contour_depMap_U, dist_contour_depMap_stretch;
	Mat label_contour_depMap, label_contour_depMap_stretch;
	//seg.morphology_proc(contour_depMap_U, contour_depMap_U_dilate, cv::MORPH_DILATE, cv::MORPH_RECT, 3, 1);
	//seg.watershed_segmentation(contour_depMap_U_dilate, markers_crop, contours, dist_contour_depMap_U, label_contour_depMap);
	seg.watershed_segmentation(contour_depMap_stretch, markers, contours, dist_contour_depMap_stretch, label_contour_depMap_stretch);

	PointCloudXYZ::Ptr input_pts(new PointCloudXYZ);
	input_pts = seg.pts_.makeShared();

	//get labels using map container
	std::map<int, pcl::PointIndices::Ptr> rough_clusters;
	int cols = label_contour_depMap_stretch.cols; int rows = label_contour_depMap_stretch.rows;
	for (int i = 0; i < rows; ++i)
	{
		if (i == 0 || i == (rows - 1))
			continue;
		for (int j = 0; j < cols; ++j)
		{
			if (j == 0 || j == cols - 1)
			{
				continue;
			}
			else
			{
				int label = label_contour_depMap_stretch.at<unsigned char>(i, j);
				if (label>0 && label != 255)
				{
					if (rough_clusters.find(label) == rough_clusters.end())//not found
					{
						pcl::PointIndices::Ptr tmp_pts_indices(new pcl::PointIndices);
						tmp_pts_indices->indices.push_back(i*cols + j);
						rough_clusters.insert(std::pair<int, pcl::PointIndices::Ptr>(label, tmp_pts_indices));

					}
					else{//found
						rough_clusters[label]->indices.push_back(i*cols + j);
					}
				}
					

			}
		}
	}

	//final_clusters is an indices vector containing the final segmentation result
	//if a cluster larger than 30000, it will be added to the under-segmentation clusters
	//else it wiil be added to final_clusters
	vector<PointCloudXYZ::Ptr> final_clusters;
	vector<pcl::PointIndices::Ptr> under_seg_clusters;
	std::map<int, pcl::PointIndices::Ptr>::iterator it_rough_clusters = rough_clusters.begin();
	for (; it_rough_clusters != rough_clusters.end(); ++it_rough_clusters)
	{
		if (it_rough_clusters->second->indices.size() < 8000
			&& it_rough_clusters->second->indices.size()>1000)//here,[ it_rough_clusters->second ] is a shared ptr of pcl::PointIndices
		{
			PointCloudXYZ::Ptr tmp_pts(new PointCloudXYZ);
			seg.extractPointsByIndices(it_rough_clusters->second, input_pts, tmp_pts,false);
			final_clusters.push_back(tmp_pts);

		}
		else{
			under_seg_clusters.push_back(it_rough_clusters->second);
		}
	}

	//viewer
	pcl::visualization::PCLVisualizer view("result");

	//first compute normal for every point
	//maybe noise filtering should be done first...
	double t1 = (double)cv::getTickCount();
	seg.integralImgNormalEstimation();
	t1 = ((double)cv::getTickCount() - t1) / cv::getTickFrequency();
	cout << "Time Consumed by Normal Estimation : " << t1 << " s" << endl;

	//PCL extract point cloud by indices(labels)
	for (vector<pcl::PointIndices::Ptr>::iterator it_under_cluster = under_seg_clusters.begin(); it_under_cluster != under_seg_clusters.end(); ++it_under_cluster)
	{
        //extract pts and remove Nan pts
		PointCloudXYZ::Ptr label_pts(new PointCloudXYZ);
		seg.extractPointsByIndices(*it_under_cluster, input_pts, label_pts,false);
        PointCloudXYZ::Ptr NanFree_pts(new PointCloudXYZ);
        vector<int> pts_index;
        pcl::removeNaNFromPointCloud<pcl::PointXYZ>(*label_pts,*NanFree_pts,pts_index);
		
        //computer normals for nan pts
        pcl::PointCloud<pcl::Normal>::Ptr NanFree_normal(new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimationOMP<pcl::PointXYZ,pcl::Normal> ne(4);
        pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        ne.setSearchMethod(tree);
        ne.setInputCloud(NanFree_pts);
        ne.setRadiusSearch(0.01);
        //ne.setKSearch(30);
        ne.compute(*NanFree_normal);

        double t1;
		t1 = (double)cv::getTickCount();
		pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
		//reg.setIndices(*it_under_cluster);
		reg.setMaxClusterSize(100000);
		reg.setMinClusterSize(100);
		reg.setSearchMethod(tree);
		reg.setNumberOfNeighbours(5);
        reg.setInputCloud(NanFree_pts);
        reg.setInputNormals(NanFree_normal);
        reg.setSmoothnessThreshold(2.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold(1);
		std::vector<pcl::PointIndices> clusters;
		reg.extract(clusters);
		t1 = ((double)cv::getTickCount() -t1) / cv::getTickFrequency();
		cout << "Time Consumed by Region Growing : " << t1 << " s" << endl;

		//after region growing, add segmented pts to final_cluster
		//for (auto& single_cluster : clusters)
		//{
		//	pcl::PointIndices::Ptr indices_ptr = boost::make_shared<pcl::PointIndices>(single_cluster);
		//	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pts(new pcl::PointCloud<pcl::PointXYZ>);
		//	seg.extractPointsByIndices(indices_ptr, label_pts, tmp_pts,false);
		//	if (tmp_pts->points.size()>1000)
		//		final_clusters.push_back(tmp_pts);
		//}

		for (std::vector<pcl::PointIndices>::iterator single_cluster = clusters.begin(); single_cluster != clusters.end();++single_cluster)
		{
			pcl::PointIndices::Ptr indices_ptr = boost::make_shared<pcl::PointIndices>(*single_cluster);
			pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_pts(new pcl::PointCloud<pcl::PointXYZ>);
            seg.extractPointsByIndices(indices_ptr, NanFree_pts, tmp_pts, false);
            if (tmp_pts->points.size()>3000)
				final_clusters.push_back(tmp_pts);
		}
	}

    vector<PointCloudXYZ::Ptr>::iterator it_final_cluster = final_clusters.begin();
	for (int i = 0; it_final_cluster != final_clusters.end(); ++it_final_cluster, ++i)
	{
        String tmp_str;
        stringstream ss;
        ss<<i;
        tmp_str = "Point cloud" + ss.str();
		pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> color(*it_final_cluster);
		view.addPointCloud<pcl::PointXYZ>(*it_final_cluster, color,tmp_str);
	}
    view.spinOnce();

    //visualize the origin pts
    pcl::visualization::PCLVisualizer view2("origin pts");
    view2.addPointCloud<pcl::PointXYZ>(seg.pts_.makeShared(),"ori pts");
    view2.spin();



	return 0;
}
