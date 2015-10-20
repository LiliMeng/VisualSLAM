#pragma once

#include <map>
#include <string>
#include <lcm/lcm.h>
#include <lcm/lcm.h>

#include "lcmtypes/mrlcm_obj_collection_t.h"
#include "lcmtypes/so_point_feature_track_list_t.h"

#include "isam/isam.h"

//Simple map class for modeling trajectory
class Map {
public:
    //Node structure used to construct the map
    typedef struct Colour_s{
        Colour_s(double r, double g, double b) :
            r(r), g(g), b(b) {
        }
        double r, g, b;
    } Colour;

    //Extends the isam Point3d to include an id for point correspondences
    class Point3d_id : public isam::Point3d {
    public:
        Point3d_id() : Point3d(), _id(0) {}
        Point3d_id(int64_t id) :
            Point3d(0.0, 0.0, 0.0), _id(id) {
        }
        Point3d_id(Point3d point, int64_t id) : Point3d(point), _id(id) {}

        Point3d_id(const Point3d_id &p) : Point3d(p), _id(p.id()) {}

        int64_t id() const {
            return _id;
        }

        void id(int64_t id){
            _id = id;
        }

    protected:
        int64_t _id;
    };

    typedef struct Node_s {
        int64_t utime;

        isam::Pose3d pose3d;
        std::vector<Point3d_id> points;
        std::vector<Colour> colours;

        double *covariance; //covariance for position [xx, xy, xz, yy, yz, zz]
    } Node;

    typedef std::vector<Node> Nodes;

    Map(void) {
    }

    Map(lcm_t *lcm) : _lcm(lcm) {

    }

    const Point3d_id getLandmark(const int64_t id);

    const Map::Node *getNode_utime(const int64_t utime);
    int num_nodes() {
        return _Nodes.size();
    }

    inline Point3d_id transformToAnchorPose(isam::Pose3d pose, Point3d_id point);
    bool augmentNode(const int64_t utime, std::vector<Point3d_id> points, std::vector<Colour> colours);
    isam::Pose3d generateNextPose(const int64_t utime);

    void setLCM(lcm_t *lcm) { _lcm = lcm; }

    void dumpMap(std::string filename);

private:
    lcm_t * _lcm;

    Nodes _Nodes;

    //The following two members provide maps for fast lookups on the Map structure
    std::map<int64_t, std::pair<int, int> > _point_feature_id_lut;  //maps feature id's to <node index, points_index> pairs within _Nodes
    std::map<int64_t, int> _node_utime_lut; //maps utimes to Node indices within _Nodes

    /// Stores the oldest pose from which a landmark can be retrieved
    /// This provides a mechanism for the caller to, in a way, reset the map. That is ,if a feature was initialized
    /// prior to this pose, then when it's id is looked up by getLandmark it will not return.
    int64_t pose_limit;
};
