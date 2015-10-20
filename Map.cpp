#include "Map.h"

using namespace std;

inline Map::Point3d_id Map::transformToAnchorPose(isam::Pose3d pose, Point3d_id point) {
    return Map::Point3d_id(pose.transform_from(point), point.id());
}

//Returns Point3D structure for landmark indexed by id
//Returns Map::Point3d_id with id = -1 if no landmarks corresponding to the id exists
const Map::Point3d_id Map::getLandmark(const int64_t id) {
    map<int64_t, pair<int, int> >::iterator offset =
        _point_feature_id_lut.find(id);

    //If not found
    if(offset == _point_feature_id_lut.end())
        return Map::Point3d_id(0, 0, 0, -1);
    else {
        return transformToAnchorPose(_Nodes[offset->second.first].pose3d,
                _Nodes[offset->second.first].points[offset->second.second]);
    }
}

// Performs naive first-order extrapolation of pose.
isam::Pose3d Map::generateNextPose(const int64_t utime){
    if (_Nodes.size() < 2)
        return isam::Pose3d(0,0,0,0,0,0);
    else
        return _Nodes.at(_Nodes.size()-1).pose3d.oplus(_Nodes.at(_Nodes.size()-1).pose3d.ominus(_Nodes.at(_Nodes.size()-2).pose3d));
}

const Map::Node *Map::getNode_utime(const int64_t utime) {
    map<int64_t, int>::iterator offset = _node_utime_lut.find(utime);

    //If not found
    if (offset == _node_utime_lut.end())
        return NULL;
    else
        return &_Nodes[offset->second];
}

//adds new points to an existing node of the map.
bool Map::augmentNode(const int64_t utime, std::vector<Point3d_id> points,
        std::vector<Colour> colours) {
    map<int64_t, int>::iterator offset = _node_utime_lut.find(utime);

    // If not found
    if(offset == _node_utime_lut.end())
        return false;
    else {
        int index = _Nodes[offset->second].points.size();
        for(unsigned int i =0; i < points.size(); i++) {
            _Nodes[offset->second].points.push_back(points[i]);
            _Nodes[offset->second].colours.push_back(colours[i]);

            //maps feature id's to <node index, points index> pairs within _Nodes
            _point_feature_id_lut[points[i].id()] = pair<int, int> (
                        offset->second, index);
            ++index;
        }
    }

    return true;
}

void Map::dumpMap(std::string filename) {
    std::ofstream out(filename.c_str());

    /// Header stuff : first line is number of nodes
    /// Second Line: space delimited string of number of points in each node
    /// M Lines: Each line stores the pose of that node
    /// M*N_{i} Lines: series of ID X Y Z lines (N_{i} lines for node i)
    out << num_nodes() <<std::endl;

    // space delimited string of number of points in each node
    for(int i=0; i<num_nodes(); i++) {
        out <<_Nodes[i].points.size() << " ";
    }
    out<< std::endl;

    // Poses
    for(int i=0; i<num_nodes(); i++) {
        out<<_Nodes[i].pose3d.x() << " "<<_Nodes[i].pose3d.y() << " "<<_Nodes[i].pose3d.z() << " "
            <<_Nodes[i].pose3d.yaw() <<" "<<_Node[i].pose3d.pitch() << " "<<_Nodes[i].pose3d.roll() << std::endl;
    }

    //Point
    for(int i=0; i<num_nodes(); i++) {
        for (unsigned int j=0; j<_Nodes[i].points.size(); j++) {
            out << _Nodes[i].points[j].id()<<" "<<_Nodes[i].points[j].x() <<" "<<_Nodes[i].points[j].y() << " "<<_Nodes[i].points[j].z()<<std::endl;
        }
    }
}
