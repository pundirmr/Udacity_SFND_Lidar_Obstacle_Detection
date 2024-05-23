/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"
#include <boost/filesystem.hpp>

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    // Add inliers
    float scatter = 0.6;
    for (int i = -5; i < 5; i++)
    {
        double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        pcl::PointXYZ point;
        point.x = i + scatter * rx;
        point.y = i + scatter * ry;
        point.z = 0;

        cloud->points.push_back(point);
    }
    // Add outliers
    int numOutliers = 10;
    while (numOutliers--)
    {
        double rx = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        double ry = 2 * (((double)rand() / (RAND_MAX)) - 0.5);
        pcl::PointXYZ point;
        point.x = 5 * rx;
        point.y = 5 * ry;
        point.z = 0;

        cloud->points.push_back(point);

    }
    cloud->width = cloud->points.size();
    cloud->height = 1;

    return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
    ProcessPointClouds<pcl::PointXYZ> pointProcessor;
    return pointProcessor.loadPcd("../../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("2D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->initCameraParameters();
    viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
    viewer->addCoordinateSystem(1.0);
    return viewer;
}

std::unordered_set<int> Ransac(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
    // Start timing the RANSAC process
    auto startTime = std::chrono::steady_clock::now();

    // Set to hold the best inliers found
    std::unordered_set<int> bestInliers;
    // Seed the random number generator
    srand(static_cast<unsigned>(time(nullptr)));

    // Lambda function to calculate the distance from a point to a line
    auto calculateDistance = [](float x1, float y1, float x2, float y2, float x3, float y3) {
        return fabs((y2 - y1) * x3 - (x2 - x1) * y3 + x2 * y1 - y2 * x1) / sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2));
    };

    // RANSAC iterations
    while (maxIterations--)
    {
        // Set to hold the inliers for the current iteration
        std::unordered_set<int> inliers;

        // Randomly pick two points to form a line
        while (inliers.size() < 2)
        {
            inliers.insert(rand() % cloud->points.size());
        }

        // Extract the points from the cloud
        auto itr = inliers.begin();
        const auto& p1 = cloud->points[*itr]; // First point
        ++itr;
        const auto& p2 = cloud->points[*itr]; // Second point

        // Iterate through all points in the cloud
        for (int i = 0; i < cloud->points.size(); ++i)
        {
            // Skip points that are already inliers
            if (inliers.count(i) > 0)
                continue;

            // Calculate the distance from the point to the line
            const auto& p3 = cloud->points[i]; // Current point
            float distance = calculateDistance(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);

            // If the distance is within the tolerance, add it to the inliers
            if (distance <= distanceTol) {
                inliers.insert(i);
            }
        }

        // Update the best inliers if the current set is better
        if (inliers.size() > bestInliers.size()) {
            bestInliers = inliers;
        }
    }

    // Stop timing the RANSAC process
    auto endTime = std::chrono::steady_clock::now();
    // Calculate and print the elapsed time
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Ransac took " << elapsedTime.count() << " milliseconds" << std::endl;

    // Return the best set of inliers found
    return bestInliers;
}


// Function to perform RANSAC for fitting a plane in a 3D point cloud
std::unordered_set<int> RansacPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
    // Start time of RANSAC execution
    auto startTime = std::chrono::steady_clock::now();

    // Set to store the indices of inliers representing the fitted plane
    std::unordered_set<int> inliersResult;

    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Iterate through RANSAC iterations
    while (maxIterations--)
    {
        // Set to store the indices of points randomly chosen to form the plane
        std::unordered_set<int> inliers;

        // Randomly select three points to form the plane
        std::uniform_int_distribution<> dis(0, cloud->points.size() - 1);
        while (inliers.size() < 3)
        {
            inliers.insert(dis(gen));
        }

        // Extract coordinates of the three points
        auto itr = inliers.begin();
        float x1 = cloud->points[*itr].x;
        float y1 = cloud->points[*itr].y;
        float z1 = cloud->points[*itr].z;
        itr++;
        float x2 = cloud->points[*itr].x;
        float y2 = cloud->points[*itr].y;
        float z2 = cloud->points[*itr].z;
        itr++;
        float x3 = cloud->points[*itr].x;
        float y3 = cloud->points[*itr].y;
        float z3 = cloud->points[*itr].z;

        // Define vectors v1 and v2 on the plane using the three points
        Eigen::Vector3f v1(x2 - x1, y2 - y1, z2 - z1);
        Eigen::Vector3f v2(x3 - x1, y3 - y1, z3 - z1);

        // Calculate coefficients of the plane equation using cross product of v1 and v2
        Eigen::Vector3f normal = v1.cross(v2);
        float a = normal.x();
        float b = normal.y();
        float c = normal.z();
        float d = -(a * x1 + b * y1 + c * z1);

        // Iterate through all points in the point cloud
        for (int index = 0; index < cloud->points.size(); index++)
        {
            // Skip if the point is already an inlier
            if (inliers.count(index) > 0)
                continue;

            // Extract coordinates of the current point
            pcl::PointXYZ point = cloud->points[index];
            float x = point.x;
            float y = point.y;
            float z = point.z;

            // Calculate the distance between the point and the plane
            float dist = fabs(a * x + b * y + c * z + d) / sqrt(a * a + b * b + c * c);

            // If the distance is within the tolerance, consider the point as an inlier
            if (dist <= distanceTol) {
                inliers.insert(index);
            }
        }

        // Update the set of inliers if the current iteration has more inliers
        if (inliers.size() > inliersResult.size()) {
            inliersResult = inliers;
        }
    }

    // End time of RANSAC execution
    auto endTime = std::chrono::steady_clock::now();

    // Calculate the elapsed time for RANSAC execution
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Output the time taken by RANSAC
    std::cout << "RansacPlane took " << elapsedTime.count() << " milliseconds" << std::endl;

    // Return the set of inliers representing the fitted plane
    return inliersResult;
}



int main()
{

    // Create viewer
    pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

    // Create data
    //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData();
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();

    // TODO: Change the max iteration and distance tolerance arguments for Ransac function
    //std::unordered_set<int> inliers = Ransac(cloud, 10, 1);

    std::unordered_set<int> inliers = RansacPlane(cloud, 10, 0.5);

    pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

    for (int index = 0; index < cloud->points.size(); index++)
    {
        pcl::PointXYZ point = cloud->points[index];
        if (inliers.count(index))
            cloudInliers->points.push_back(point);
        else
            cloudOutliers->points.push_back(point);
    }


    // Render 2D point cloud with inliers and outliers
    if (inliers.size())
    {
        renderPointCloud(viewer, cloudInliers, "inliers", Color(0, 1, 0));
        renderPointCloud(viewer, cloudOutliers, "outliers", Color(1, 0, 0));
    }
    else
    {
        renderPointCloud(viewer, cloud, "data");
    }

    while (!viewer->wasStopped())
    {
        viewer->spinOnce();
    }

}
