#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py = pybind11;

using namespace std;
using namespace cv;
using namespace g2o;

class EdgeProjection : public g2o::BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjection(const Eigen::Matrix<double, 4, 4> &K,const Eigen::Matrix<double, 4, 9> &Cor,const int index ) : _K(K),_Cor(Cor),_index(index) {}

  virtual void computeError() override {
    // P3d----vertex0,T----vertex1
    const VertexSBAPointXYZ *v0 = static_cast<const VertexSBAPointXYZ *> (_vertices[0]);
    const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
    Eigen::Vector3d Di(v0->estimate());
    Eigen::Vector4d Di_linear ;
    Di_linear << Di(0),Di(1),Di(2),1.0;
    Eigen::Matrix4d diagDi(Di_linear.asDiagonal());
    // Eigen::Matrix<double, 4, 4> T;
    g2o::SE3Quat T_quat=v1->estimate();
    g2o::Matrix4D T= T_quat.to_homogeneous_matrix();
    Eigen::Matrix<double, 4, 9> pos_pixel = _K  * T *diagDi * _Cor;
    Eigen::Matrix<double, 2, 1> optimize;
    pos_pixel.row(0) = pos_pixel.row(0).cwiseQuotient(pos_pixel.row(2));
    pos_pixel.row(1) = pos_pixel.row(1).cwiseQuotient(pos_pixel.row(2));
    // cout << "_measurement"<< endl<<_measurement<< endl;
    optimize << pos_pixel(0,_index),pos_pixel(1,_index);
    // cout << "optimize"<< endl<<optimize<< endl;
    Vector2D obs(_measurement);
    _error = obs- optimize;
    // cout << "_error"<< endl<<_error<< endl;
  }
  // virtual void linearizeOplus(){};
  virtual void linearizeOplus() override  {
    VertexSE3Expmap * vj = static_cast<VertexSE3Expmap *>(_vertices[1]);
    SE3Quat T(vj->estimate());
    // vi是维度向量Di
    VertexSBAPointXYZ* vi = static_cast<VertexSBAPointXYZ*>(_vertices[0]);
    Eigen::Vector3d Di = vi->estimate();
    // 把维度转化成车辆的角点
    Eigen::Vector3d xyz (Di(0,0)*_Cor(0,_index),Di(1,0)*_Cor(1,_index),Di(2,0)*_Cor(2,_index));
    Eigen::Vector3d xyz_trans = T.map(xyz);
    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    double z_2 = z*z;

    Eigen::Matrix<double,2,3,Eigen::ColMajor> tmp;
    tmp(0,0) = _K(0,0);
    tmp(0,1) = 0;
    tmp(0,2) = -x/z*_K(0,0);

    tmp(1,0) = 0;
    tmp(1,1) = _K(0,0);
    tmp(1,2) = -y/z*_K(0,0);
    // e_cp对维度求导
    for(int vertex=0;vertex<9;vertex++){
      Eigen::Vector3d corner_col((_Cor.col(vertex)).topRows(3));
      Eigen::Matrix3d diagCor_col(corner_col.asDiagonal());
      _jacobianOplusXi +=  -1./z * tmp * T.rotation().toRotationMatrix() * diagCor_col;  
    }
    _jacobianOplusXi = _jacobianOplusXi / 9;
    // e_cp对位姿李代数求导
    _jacobianOplusXj(0,0) =  x*y/z_2 *_K(0,0);
    _jacobianOplusXj(0,1) = -(1+(x*x/z_2)) *_K(0,0);
    _jacobianOplusXj(0,2) = y/z *_K(0,0);
    _jacobianOplusXj(0,3) = -1./z *_K(0,0);
    _jacobianOplusXj(0,4) = 0;
    _jacobianOplusXj(0,5) = x/z_2 *_K(0,0);

    _jacobianOplusXj(1,0) = (1+y*y/z_2) *_K(0,0);
    _jacobianOplusXj(1,1) = -x*y/z_2 *_K(0,0);
    _jacobianOplusXj(1,2) = -x/z *_K(0,0);
    _jacobianOplusXj(1,3) = 0;
    _jacobianOplusXj(1,4) = -1./z *_K(0,0);
    _jacobianOplusXj(1,5) = y/z_2 *_K(0,0);
}
  virtual bool read(istream &in) {}

  virtual bool write(ostream &out) const {}

private:

  Eigen::Matrix<double, 4, 9> _Cor;
  Eigen::Matrix<double, 4, 4> _K;
  int _index;
};

/**
    input1:[w h l ry3d cx3d cy3d cz]
    input2:[x0,y0,x1,y1,...]
    input3:[f,cx,cy]  
*/
py::array_t<double> optimize(py::array_t<double>& input1, py::array_t<double>& input2,py::array_t<double>& input3,
                             py::array_t<double>& score,py::array_t<double>& hm_score) {
    // 获取input的信息
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();
    py::buffer_info buf3 = input3.request();
    py::buffer_info buf_score = score.request();
    py::buffer_info buf_hm_score = hm_score.request();
    if (buf1.ndim !=1 || buf2.ndim !=1 || buf3.ndim !=1 || buf_score.ndim !=1 || buf_hm_score.ndim !=1 ){
        throw std::runtime_error("Number of dimensions must be one");
    }
    if (buf1.size !=7){
        throw std::runtime_error("Input 3d bbox must match the format [w h l ry3d cx3d cy3d cz]");
    }
    if (buf2.size !=18){
        throw std::runtime_error("Input keypoints must match the format [x0,y0,x1,y1,...]");
    }
    if (buf3.size !=3){
        throw std::runtime_error("Input camera intrinsics must match the format [f,cx,cy]");
    }
    if (buf_score.size !=1){
        throw std::runtime_error("Input 2D bbox center score");
    }
    if (buf_hm_score.size !=9){
        throw std::runtime_error("Input 9 keypoints score");
    }
    //获取numpy.ndarray 数据指针
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr3 = (double*)buf3.ptr;
    double* ptr_score = (double*)buf_score.ptr;
    double* ptr_hm_score = (double*)buf_hm_score.ptr;
    double w_3d = ptr1[0], h_3d = ptr1[1], l_3d = ptr1[2]; 
    double ry3d = ptr1[3];
    double cx_3d = ptr1[4], cy_3d = ptr1[5], c_z = ptr1[6];
    Eigen::Matrix<double, 4, 9> Cor;//l,h,w
    Cor<< 1./2, 1./2, -1./2, -1./2, 1./2, 1./2, -1./2, -1./2, 0,
        0,0,0,0,-1,-1,-1,-1,-1./2,
        1./2, -1./2, -1./2, 1./2, 1./2, -1./2, -1./2, 1./2, 0,
        1,1,1,1,1,1,1,1,1;
    cout<<"优化前:"<<endl;
    cout << "角度 = " << ry3d << endl;
    cout<<"3D bbox中心点坐标：\t cx:"<< cx_3d << "\t cy:" <<
    cy_3d<< "\t cz:"  << c_z  <<endl;
    cout<<"3D bbox长宽高lhw : \t L: "<< l_3d<<"\t H: "<<h_3d<<"\t W: "<<w_3d<<endl;
    Eigen::Matrix<double, 18, 1> keypoints_2d;
    keypoints_2d.fill(0);
    for(int i=0;i<18;i++){
        keypoints_2d[i] = ptr2[i]; 
    }
    

    // 初始化g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3

    //Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    std::unique_ptr<Block::LinearSolverType> linearSolver ( new g2o::LinearSolverCSparse<Block::PoseMatrixType>());

    //Block* solver_ptr = new Block ( linearSolver );
    //std::unique_ptr<Block> solver_ptr ( new Block ( linearSolver));
    std::unique_ptr<Block> solver_ptr ( new Block ( std::move(linearSolver)));     // 矩阵块求解器

    //g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr));

    g2o::SparseOptimizer optimizer;

    optimizer.setAlgorithm ( solver );
    
    // pose对应转移矩阵
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    // 用theta对应的旋转矩阵初始化R_mat,c3d初始化平移向量t
    Eigen::Matrix3d R_mat;
    Eigen::Vector3d t_3d;
    // 内参矩阵
    Eigen::Matrix4d K_Mat ;
    K_Mat << ptr3[0], 0,         ptr3[1],  0,
            0,        ptr3[0],   ptr3[2],  0,
            0,        0,         1.0,      0,
            0,        0,         0,        1.0;
    R_mat<< +cos(ry3d), 0, +sin(ry3d),
            0, 1, 0,
            -sin(ry3d), 0, cos(ry3d);
    t_3d<< cx_3d, cy_3d, c_z;
    // 把转移矩阵添加至图顶点
    pose->setId ( 0 );
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            t_3d
                        ));
    optimizer.addVertex ( pose );

    // 把Di当成第二个顶点
    g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
    point->setId ( 1 );
    point->setEstimate ( Eigen::Vector3d (l_3d,h_3d,w_3d));
    point->setMarginalized ( true ); 
    optimizer.addVertex ( point );

    // parameter: camera intrinsics
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        ptr3[0], Eigen::Vector2d (ptr3[1], ptr3[2] ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    int index = 1;
    for(int i=0;i<9;i++){
        EdgeProjection *edge = new EdgeProjection(K_Mat,Cor,i);
        edge->setId ( index );
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( 1 ) ) );
        edge->setVertex ( 1, pose );
        edge->setMeasurement ( Eigen::Vector2d (keypoints_2d(i),keypoints_2d(i+9)));
        edge->setParameterId ( 0,0 );
        Eigen::Vector2d confidence(ptr_hm_score[i],ptr_score[0]);
        Eigen::Matrix2d conviance_matrix(confidence.asDiagonal());
        edge->setInformation ( conviance_matrix );
        optimizer.addEdge ( edge ); 
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"优化过程耗时: "<<time_used.count() <<" 秒."<<endl;
    cout<<"优化后:"<<endl;
    Eigen::Isometry3d T( pose->estimate() );
    Eigen::Vector3d euler_angles =pose->estimate().rotation().toRotationMatrix().eulerAngles( 0, 1, 2 );
    cout << "角度 = " << euler_angles(1) << endl;
    cout<<"3D bbox中心点坐标：\t cx:"<< T.matrix()(0,3) << "\t cy:" <<
    T.matrix()(1,3)<< "\t cz:"  <<T.matrix()(2,3)  <<endl;
    cout<<"3D bbox长宽高lhw: \t L: "<< point->estimate()(0)<<"\t H: "<<point->estimate()(1)<<"\tW: "<<point->estimate()(2)<<endl;
    
    auto result = py::array_t<double>(7);
    py::buffer_info buf4 = result.request();
    double* ptr4 = (double*)buf4.ptr;
    ptr4[0] = point->estimate()(2);//w
    ptr4[1] = point->estimate()(1);//h
    ptr4[2] = point->estimate()(0);//l
    ptr4[3] = euler_angles(1);     //ry3d
    ptr4[4] = T.matrix()(0,3);     //cx
    ptr4[5] = T.matrix()(1,3);     //cy
    ptr4[6] = T.matrix()(2,3);     //cz
    
    return result;
}

PYBIND11_MODULE(energy, m) {

    m.doc() = "optimize using numpy and g2o!";

    m.def("optimize", &optimize);
}