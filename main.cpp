#include <iostream>
#include <vector>
#include <cmath>

#include <map>
#include <fstream>
#include <sstream>

#include <Eigen/SVD> 
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/*
  SinglePoint DataStructure can either query as x, y, z or using data[0], data[1], data[2]
*/

struct point3d_s {
  union {
    double data[3];
    struct {
      double x;
      double y;
      double z;
    };
  };
};

typedef std::vector<point3d_s> point3d_list_s;
typedef std::map<int, int> correspondences_s;
typedef Eigen::Matrix4f    transformation_s;

//NOTE: Not using pointcloud_s data structure, will typedef vector of point3d as pointslist;
//NOTE: Not using this data structure and using std::pair<int, int> as  correspondences data structure
/*struct correspondences_s {};*/

//NOTE: Not using transformation_s data structure. Will be using Eigen for transofrmation representation
/*struct transformation_s {}; */

class PointCloud
{
  point3d_list_s points;
  int width_ = 1;
  int height_ = 0;

public:
  PointCloud() = default;
  
  void push_back(point3d_s point) {
    points.push_back(point);
    height_ ++;
  }
  
  void print() {
    printf("Width: %d, Height: %d \n", width_, height_);
    for (auto it = points.begin(); it != points.end(); it++) {
      printf("%0.4f %0.4f %0.4f \n", it->x, it->y, it->z);
    }   
  }

  // accessors
  int height() const {return height_;}
  int width()  const {return width_;};
  
  // Operator Overloading functions to retreive data rather than querying points
  // Can query each point as point_cloud[idx].x point_cloud[idx].y point_cloud[idx].z
  // OR query as point_cloud[idx][0]  point_cloud[idx][1] point_cloud[idx][2]
  // OR each point structure
  point3d_s operator [] (int idx) const {
    return points[idx];
  }
  
};
// function that loads point cloud from text file
void load_data(std::string file_name, PointCloud &cloud)
{
  ifstream file(file_name);
  string line;
  while (getline(file, line)) {
    stringstream ss(line);

    point3d_s point;
    
    ss >> point.x >> point.y >> point.z;
    cloud.push_back(point);
  }
  file.close();
}

//function that loads correspondences from txt file
void load_correspondences(std::string filename, correspondences_s &correspondences)
{
  ifstream file(filename);
  string line;
  while(getline(file, line)) {
    stringstream ss(line);
    int feature_idx_1, feature_idx_2;

    ss >> feature_idx_1 >> feature_idx_2;
    correspondences[feature_idx_1] = feature_idx_2;    
  }
  file.close();
}

point3d_s compute_centroid_pointcloud (const PointCloud &src) {
  point3d_s mean{0.0, 0.0, 0.0};
  
  for (int i = 0; i < src.height(); i++) {
    mean.x += src[i].x;
    mean.y += src[i].y;
    mean.z += src[i].z;
  }
  for (int i = 0; i < 3; i++) {
    mean.data[i] /= src.height();
  }
  return mean;
}

MatrixXf mean_shift_pointcloud(const PointCloud &src, const point3d_s &mean,
			       correspondences_s correspondences, bool useCorresponding=false) {

  Eigen::MatrixXf output(3, src.height());

  for (int i = 0; i < src.height(); i++) {
    int idx = i;
    if (useCorresponding) {
      idx = correspondences[i];
    }
    output(0, i) = src[idx].x - mean.x;
    output(1, i) = src[idx].y - mean.y;
    output(2, i) = src[idx].z - mean.z;
  }
  return output;
}

// function that estimates transform with known correspondences
transformation_s estimate_transformation(const PointCloud &src, const PointCloud &dst,
					 const correspondences_s &correspondences)
{
  /*Find the mean of each pointcloud.
    Move points relative to the mean of point cloud
    Populate the points in Eigen Matris and compute Covariance Matrix
    Computee USV = SVD(covariance_matrix)
    Rotation = V * U_transpose
    Translation = dst - Rotation*src; 
   */
  transformation_s transformation;

  point3d_s mean_src = compute_centroid_pointcloud(src);
  point3d_s mean_dst = compute_centroid_pointcloud(dst);

  Eigen::MatrixXf mean_src_matrix(3, 1);
  Eigen::MatrixXf mean_dst_matrix(3, 1);

  mean_src_matrix << mean_src.x, mean_src.y, mean_src.z;
  mean_dst_matrix << mean_dst.x, mean_dst.y, mean_dst.z;

  Eigen::MatrixXf meanshifted_pointcloud_src = mean_shift_pointcloud(src, mean_src, correspondences);
  Eigen::MatrixXf meanshifted_pointcloud_dst = mean_shift_pointcloud(dst, mean_dst, correspondences, true);
  //Now the pointclouds indices are one to one corresponding to their indices

  Eigen::MatrixXf covariance = meanshifted_pointcloud_src * meanshifted_pointcloud_dst.transpose();
 

  Eigen::JacobiSVD<MatrixXf> svd(covariance, ComputeFullU | ComputeFullV);
  Eigen::MatrixXf rotation = svd.matrixV() * svd.matrixU().transpose();
  transformation.block<3, 3>(0, 0) = rotation; 
  transformation.block<3, 1>(0, 3) = -rotation*mean_src_matrix + mean_dst_matrix;
  transformation(3,3) = 1;

  return transformation;
}


// TODO: function that transforms a pointcloud (pc) given a 6DOF transformation. pc_out is the transformed pointcloud
void transform_pointcloud(const PointCloud &pc, PointCloud &pc_out, transformation_s transformation)
{
  /*
    Reduced space transformation. Computationally takes same time to dump all points into Eigen and 
    transform or transform each point
   */
  for (int i = 0; i < pc.height(); i++) {
    Eigen::MatrixXf point(4, 1);
    point(0, 0) = pc[i].x;
    point(1, 0) = pc[i].y;
    point(2, 0) = pc[i].z;
    point(3, 0) = 1;

    Eigen::MatrixXf transformed_point = transformation*point;
    point3d_s tx_point;
    tx_point.x = transformed_point(0, 0);
    tx_point.y = transformed_point(1, 0);
    tx_point.z = transformed_point(2, 0);

    pc_out.push_back(tx_point);
  }
}

// TODO: function that prints transformation
void print_transform(const transformation_s &transform)
{
 std::cout << transform << std::endl;
}

int main() {
    PointCloud pc1, pc2;
    //load point cloud data
    load_data("../pc1.txt", pc1);
    load_data("../pc2.txt", pc2);

    correspondences_s correspondences;
    //load correspondences
    load_correspondences("../correspondences.txt", correspondences);

    //estimate transformation (source = pc1, destination = pc2)
    transformation_s est_transformation = estimate_transformation(pc1,pc2,correspondences);
    // print out the result transformation
    print_transform(est_transformation);
    PointCloud transformed_pc;
    transform_pointcloud(pc1, transformed_pc, est_transformation);

    return 0;
}
