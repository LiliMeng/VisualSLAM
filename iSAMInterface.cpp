#include "iSAMInterface.h"

iSAMInterface::iSAMInterface()
{
    Eigen::Matrix4f M;
    M<< 0, 0, 1, 0,
       -1, 0, 0, 0,
       0, -1, 0, 0,
        0, 0, 0, 1;

    transformation2isam = M;

    Eigen::Matrix<double, 6, 6> N;
    N<<
      0, 0, 1, 0, 0, 0,
     -1, 0, 0, 0, 0, 0,
      0,-1, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 1, 0,
      0, 0, 0, 1, 0, 0;

    covariance2isam = N;

    _slam = new isam::Slam;

//  isam::Properties prop;
//  prop.method = isam::LEVENBERG_MARQUARDT;
//  _slam->set_properties(prop);
}
#include "iSAMInterface.h"

iSAMInterface::~iSAMInterface()
{
    delete _slam;

    for(std::map<uint64_t, isam::Pose3d_Node *>::iterator it = _camera_nodes.begin(); it !=_camera_nodes.end(); ++it)
    {
        delete it->second;
    }
}

void iSAMInterface::addCameraCameraConstraint(uint64_t time1, uint64_t time2,
                                        const Eigen::Matrix3f & Rprev, const Eigen::Vector3f & tprev,
                                        const Eigen::Matrix3f & Rcurr, const Eigen::Vector3f & tcurr,
                                        Eigen::MatrixXd & cov)
{
    std::pair<uint64_t, uint64_t> constraint(time1, time2);

    if(time1 > time2 || cameraCameraConstraint[constraint])
    {
        return;
    }

    cameraCameraConstraints[constraint] = true;

    Eigen::Matrix4f prev = Eigen::Matrix4f::Identity();
    prev.topLeftCorner(3,3) = Rprev;
    prev.topRightCorner(3, 1) = tprev;

    Eigen::Matrix4f curr = Eigen::Matrix4f::Identity();
    curr.topLeftCorner(3,3) = Rcurr;
    curr.topRightCorner(3,1) = tcurr;

    Eigen::Matrix4f eigenDelta = prev.inverse() * curr;

    Eigen::Matrix4f d = transformation2isam * eigenDelta * transformation2isam.inverse();

    Eigen::Matrix3f rot = d.topLeftCorner(3, 3);
    Eigen::Quaternionf quat(rot);
    Eigen::Vector3f trans = d.topRightCorner(3, 1);

    isam::Pose3d delta(isam::Point3d(trans(0), trans(1), trans(2)), Eigen::Quaterniond(quat.w(), quat.x(), quat.y(), quat.z()));

    Eigen::MatrixXd covIsam = covariance2isam * cov * covariance2isam.inverse();

    isam::Pose3d_Node* node1 = cameraNode(time1);
    isam::Pose3d_Node* node2 = cameraNode(time2);
    isam::Pose3d_Pose3d_Factor * factor = new isam::Pose3d_Pose3d_Factor(node1, node2, delta, isam::Covariance(covIsam));

    _slam->add_factor(factor);
}

void iSAMInterface::addLoopConstraint(uint64_t time1, uint64_t time2, Eigen::Matrix4d & loopConstraint, Eigen::MatrixXd & cov)
{
    std::pair<uint64_t, uint64_t> constraint(time1, time2);

    assert(!cameraCameraConstraint[constraint]);

    assert(_camera_nodes[time1]);

    assert(_camera_nodes[time2]);

    isam::Pose3d delta(transformation2isam.cast<double>().eval() * loopConstraint * transformation2isam.cast<double>().eval().inverse());

    isam::Pose3d_Node * nodeTime1 = cameraNode(time1);
    isam::Pose3d_Node * nodeTime2 = cameraNode(time2);

    Eigen::MatrixXd covIsam = covariance2isam * cov * covariance2isam.inverse();

    isam::Pose3d_Pose3d_Factor * factor = new isam::Pose3d_Pose3d_Factor(nodeTime1, nodeTime2, delta, isam::Covariance(covIsam));

    _slam->add_factor(factor);
}

isam::Pose3d_Node * iSAMInterface::cameraNode(uint64_t time)
{
    std::map<uint64_t, isam::Pose3d_Node *>::iterator it = _camera_nodes.find(time);

    if(it == _camera_nodes.end())
    {
        isam::Pose3d_Node * new_pose_node = new isam::Pose3d_Node();

        _camear_nodes[time] = new_pose_node;

        _slam->add_node(new_pose_node);

        if(_camera_nodes.size() ==1)
        {
            isam::Pose3d_Factor * prior = new isam::Pose3d_Factor(new_pose_node,
                                                                  isam::Pose3d(),
                                                                  isam::Covariance(0.001 * Eigen::Matrix<double, 6, 6>::Identity()));
            _slam->add_factor(prior);

        }

        return new_pose_node;
    }
    else
    {
        return it->second;
    }
}

void iSAMInterface::optimise()
{
    _slam->batch_optimization();
}

const std::list<isam::Factor* > & iSAMInterface::getFactors()
{
    return _slam->get_factors();
}

void iSAMInterface::getCameraPoses(std::vector<std::pair<uint64_t, Eigen::Matrix4f> > & poses)
{
    std::pair<uint64_t, Eigen::Matrix4f> next;

    for(std::map<uint64_t, isam::Pose3d_Node *>::iterator it = _camera_nodes.begin(); it != _camera_nodes.end(); ++it)
    {
        Eigen::Matrix4f inIsam = it->second->value().wTo.cast<float>();

        next.first = it->first;
        next.second = transformation2isam.inverse() * inIsam * transformation2isam;

        poses.push_back(next);
    }
}

Eigen::Matrix4f iSAMInterface::getCameraPose(uint64_t id)
{
    Eigen::Matrix4f inIsam = _camera_nodes[id]->value().wTo().cast<float>();
    Eigen::Matrix4f inWorld = transformation2isam.inverse() * inIsam * transformation2isam;
    return inWorld;
}


