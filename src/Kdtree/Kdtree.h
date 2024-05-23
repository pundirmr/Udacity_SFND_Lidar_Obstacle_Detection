#ifndef KDTREE_H
#define KDTREE_H

#include <vector>
#include <cmath>

// Structure to represent a node in the KD-Tree
struct Node {
    std::vector<float> point;
    int id;
    Node* left;
    Node* right;

    Node(const std::vector<float>& arr, int setId)
        : point(arr), id(setId), left(nullptr), right(nullptr) {}
};

// KD-Tree class for processing 3D points
class KdTree {
public:
    KdTree() : root(nullptr) {}

    // Insert a new point into the KD-Tree
    void insert(const std::vector<float>& point, int id) {
        insertHelper(root, 0, point, id);
    }

    // Return a list of point ids in the tree that are within distance of the target
    std::vector<int> search(const std::vector<float>& target, float distanceTol) {
        std::vector<int> ids;
        searchHelper(target, root, 0, distanceTol, ids);
        return ids;
    }

private:
    Node* root;

    // Helper function to insert a new point into the KD-Tree
    void insertHelper(Node*& node, unsigned int depth, const std::vector<float>& point, int id) {
        if (node == nullptr) {
            node = new Node(point, id);
        }
        else {
            unsigned int dimension = depth % 3;
            if (point[dimension] < node->point[dimension]) {
                insertHelper(node->left, depth + 1, point, id);
            }
            else {
                insertHelper(node->right, depth + 1, point, id);
            }
        }
    }

    // Helper function to search for points within a certain distance of the target
    void searchHelper(const std::vector<float>& target, Node* node, unsigned int depth, float distanceTol, std::vector<int>& ids) {
        if (node == nullptr) return;

        if (isPointWithinBox(node->point, target, distanceTol)) {
            if (calculateDistance(node->point, target) <= distanceTol) {
                ids.push_back(node->id);
            }
        }

        unsigned int dimension = depth % 3;
        if (target[dimension] - distanceTol < node->point[dimension]) {
            searchHelper(target, node->left, depth + 1, distanceTol, ids);
        }
        if (target[dimension] + distanceTol > node->point[dimension]) {
            searchHelper(target, node->right, depth + 1, distanceTol, ids);
        }
    }

    // Check if a point is within the bounding box defined by the target and distance tolerance
    bool isPointWithinBox(const std::vector<float>& point, const std::vector<float>& target, float distanceTol) const {
        return (point[0] >= target[0] - distanceTol && point[0] <= target[0] + distanceTol) &&
            (point[1] >= target[1] - distanceTol && point[1] <= target[1] + distanceTol) &&
            (point[2] >= target[2] - distanceTol && point[2] <= target[2] + distanceTol);
    }

    // Calculate the Euclidean distance between two points
    float calculateDistance(const std::vector<float>& point1, const std::vector<float>& point2) const {
        return std::sqrt(std::pow(point1[0] - point2[0], 2) +
            std::pow(point1[1] - point2[1], 2) +
            std::pow(point1[2] - point2[2], 2));
    }
};

#endif // KDTREE_H
