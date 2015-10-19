#ifndef ISAMINTERFACE_H_
#define ISAMINTERFACE_H_

#include <string>
#include <map>
#include <Eigen/Dense>
#include <isam/isam.h>
#include <stdint.h>

class iSAMInterface
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    iSAMInterface();
    virtual ~iSAMInterface();

    void addCameraCameraConstraint(uint64_t time1, uint64_t time2,
                                   const Eigen::Matrix3f & Rprev, const Eigen::Vector3f & tprev,
                                   const Eigen::Matrix3f & Rcurr, const Eigen::Vector3f & tcurr,
                                   Eigen::MatrixXd & cov);

    void addLoopConstraint(uint64_t time1,
                           uint64_t time2,
                           Eigen::Matrix4d & loopConstraint,
                           Eigen::MatrixXd & cov);

    const std::list<isam::Factor* > & getFactors();

    void getCameraPoses(std::vector<std::pair<uint64_t, Eigen::Matrix4f> > & poses);

    void optimise();

    Eigen::Matrix4f getCameraPose(uint64_t id);

private:
    isam::Pose3d_Node * cameraNode(uint64_t time);

    isam::Slam * _slam;
    std::map<uint64_t, isam::Pose3d_Node*> _camera_nodes;

    Eigen::Matrix4f transformation2isam;
    Eigen::MatrixXd covariance2isam;
    std::map<std::pair<uint64_t, uint64_t>, bool> cameraCameraConstraints;

};


#endif /* ISAMINTERFACE_H_ */
