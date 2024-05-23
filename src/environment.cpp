/* \author Aaron Brown */
// Create simple 3d highway enviroment using PCL
// for exploring self-driving car sensors

#include "sensors/lidar.h"
#include "render/render.h"
#include "processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "processPointClouds.cpp"

/// <summary>
/// Initialise highway
/// </summary>
/// <param name="renderScene"></param>
/// <param name="viewer"></param>
/// <returns></returns>
std::vector<Car> initHighway(bool renderScene, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    Car egoCar( Vect3(0,0,0), Vect3(4,2,2), Color(0,1,0), "egoCar");
    Car car1( Vect3(15,0,0), Vect3(4,2,2), Color(0,0,1), "car1");
    Car car2( Vect3(8,-4,0), Vect3(4,2,2), Color(0,0,1), "car2");	
    Car car3( Vect3(-12,4,0), Vect3(4,2,2), Color(0,0,1), "car3");
  
    std::vector<Car> cars;
    cars.push_back(egoCar);
    cars.push_back(car1);
    cars.push_back(car2);
    cars.push_back(car3);

    if(renderScene)
    {
        renderHighway(viewer);
        egoCar.render(viewer);
        car1.render(viewer);
        car2.render(viewer);
        car3.render(viewer);
    }

    return cars;
}

/// <summary>
/// Open 3D viewer and display simple highway
/// </summary>
/// <param name="viewer"></param>
void simpleHighway(pcl::visualization::PCLVisualizer::Ptr& viewer)
{
    // Render options
    const bool renderScene = false;
    const bool renderPoints = true;
    const bool renderClusters = true;
    const bool renderBoxes = true;

    // Initialize highway environment
    std::vector<Car> cars = initHighway(renderScene, viewer);

    // Create Lidar sensor
    const double groundSlope = 0;
    auto lidar = std::make_unique<Lidar>(cars, groundSlope);
    auto pointCloud = lidar->scan();

    // Create point processor
    ProcessPointClouds<pcl::PointXYZ> pointProcessor;

    // Segment the point cloud into road and obstacles
    auto segmentCloud = pointProcessor.SegmentPlane(pointCloud, 100, 0.2);

    if (renderPoints) {
        renderPointCloud(viewer, segmentCloud.first, "obstCloud", Color(1, 0, 0));   // obstacle cloud
        renderPointCloud(viewer, segmentCloud.second, "planeCloud", Color(0, 1, 0)); // road cloud
    }

    // Cluster the obstacle cloud
    auto cloudClusters = pointProcessor.Clustering(segmentCloud.first, 1.5, 3, 30);

    // Define colors for the clusters
    std::vector<Color> colors = { Color(1, 0, 0), Color(0, 1, 0), Color(0, 0, 1) };

    int clusterId = 0;
    for (const auto& cluster : cloudClusters) {
        if (renderClusters) {
            std::cout << "cluster size ";
            pointProcessor.numPoints(cluster);
            renderPointCloud(viewer, cluster, "obstCloud" + std::to_string(clusterId), colors[clusterId % colors.size()]);
        }

        if (renderBoxes) {
            Box box = pointProcessor.BoundingBox(cluster);
            renderBox(viewer, box, clusterId, colors[clusterId % colors.size()], 0.7);
        }

        ++clusterId;
    }
}

void renderBoxOnBounds(pcl::visualization::PCLVisualizer::Ptr& viewer, int clusterId, float xmin, float ymin, float zmin, float xmax, float ymax, float zmax, bool display)
{
    if (display)
    {
        Box box;
        box.x_min = xmin;
        box.y_min = ymin;
        box.z_min = zmin;
        box.x_max = xmax;
        box.y_max = ymax;
        box.z_max = zmax;
        renderBox(viewer, box, clusterId, Color(1, 0, 1), 0.6);
    }
}

/// <summary>
/// Load multiple files and view city block
/// </summary>
/// <param name="viewer"></param>
/// <param name="pointProcessorI"></param>
/// <param name="inputCloud"></param>
void cityBlockStream(pcl::visualization::PCLVisualizer::Ptr& viewer, ProcessPointClouds<pcl::PointXYZI>* pointProcessorI, const pcl::PointCloud<pcl::PointXYZI>::Ptr& inputCloud)
{
    const bool showInputCloud = false;
    const bool showFilteredCloud = true;
    const bool showVanBox = false;
    const bool useCustomImplementation = true;
    const float distanceTol = 0.2f; // in meters
    const int minClusterSize = 50;
    const int maxClusterSize = 2000;
    const std::vector<Color> colors = { Color(0, 1, 1), Color(1, 0.92, 0.016), Color(0, 0, 1) };

    if (showInputCloud) {
        renderPointCloud(viewer, inputCloud, "inputCloud");
        return;
    }

    auto filterCloud = pointProcessorI->FilterCloud(inputCloud, 0.1f, Eigen::Vector4f(-20.0f, -5.0f, -2.0f, 1), Eigen::Vector4f(20.0f, 8.0f, 2.0f, 1));
    if (showFilteredCloud) {
        renderPointCloud(viewer, filterCloud, "filterCloud");
    }

    // Render box 
    int clusterId = 0;
    bool display = false;
    bool boundsV1 = false;
    float xmin = -1.5;
    float ymin = -1.7;
    float zmin = -1;
    float xmax = 2.6;
    float ymax = 1.7;
    float zmax = 0.4;

    if (!boundsV1)
    {
        xmin = -10;
        ymin = -5;
        zmin = -5;
        xmax = 30;
        ymax = 8;
        zmax = 1;
    }   
    renderBoxOnBounds(viewer, clusterId, xmin, ymin, zmin, xmax, ymax, zmax, false);
    clusterId++;

    // Separate the point cloud into planes - road and obstacles
    ProcessPointClouds<pcl::PointXYZI> pointProcessor;
    std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> segmentCloud;

    if (!useCustomImplementation) {
        segmentCloud = pointProcessor.SegmentPlane(filterCloud, 5, 0.4);
    }
    else {
        std::cout << "Segmenting plane using RANSAC" << std::endl;
        segmentCloud = pointProcessor.SegmentPlaneWithRansac(filterCloud, 5, 0.4);
    }

    renderPointCloud(viewer, segmentCloud.first, "obstacleCloud", Color(1, 0, 0)); // Obstacle cloud
    renderPointCloud(viewer, segmentCloud.second, "roadCloud", Color(0, 1, 0));    // Road cloud

    // Cluster the obstacles to differentiate between various objects
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloudClusters;

    if (!useCustomImplementation) {
        std::cout << "Performing PCL Clustering" << std::endl;
        cloudClusters = pointProcessor.Clustering(segmentCloud.first, distanceTol, minClusterSize, maxClusterSize);
    }
    else {
        std::cout << "Performing Euclidean Clustering" << std::endl;
        cloudClusters = pointProcessor.EuclideanClustering(segmentCloud.first, distanceTol, minClusterSize, maxClusterSize);
    }

    for (const auto& cluster : cloudClusters) {
        renderPointCloud(viewer, cluster, "obstacleCloud" + std::to_string(clusterId), colors[clusterId % colors.size()]);

        Box box = pointProcessor.BoundingBox(cluster);
        renderBox(viewer, box, clusterId, Color(1, 1, 1), 0.7);

        ++clusterId;
    }
}

/// <summary>
/// Loads a single pcd file and view
/// </summary>
/// <param name="viewer"></param>
void cityBlockSingleFile(pcl::visualization::PCLVisualizer::Ptr& viewer)
{
    // Create a point processor for point clouds with intensity values
    ProcessPointClouds<pcl::PointXYZI> pointProcessorI;
    pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloud = pointProcessorI.loadPcd("../../src/sensors/data/pcd/data_1/0000000000.pcd");
    pcl::PointCloud<pcl::PointXYZI>::Ptr filterCloud = pointProcessorI.FilterCloud(inputCloud, 0.1f, Eigen::Vector4f(-20.0f, -5.0f, -2.0f, 1), Eigen::Vector4f(20.0f, 8.0f, 2.0f, 1));

    // Render box 
    int clusterId = 0;
    bool display = false;
    bool boundsV1 = false;
    float xmin = -1.5;
    float ymin = -1.7;
    float zmin = -1;
    float xmax = 2.6;
    float ymax = 1.7;
    float zmax = 0.4;

    if (!boundsV1)
    {
        xmin = -10;
        ymin = -5;
        zmin = -5;
        xmax = 30;
        ymax = 8;
        zmax = 1;
    }
    renderBoxOnBounds(viewer, clusterId, xmin, ymin, zmin, xmax, ymax, zmax, false);
    clusterId++;

    // Segment the point cloud into road and obstacles
    auto segmentCloud = pointProcessorI.SegmentPlane(filterCloud, 100, 0.2);

    renderPointCloud(viewer, segmentCloud.first, "obstacleCloud", Color(1, 0, 0)); // obstacle cloud
    renderPointCloud(viewer, segmentCloud.second, "roadCloud", Color(0, 1, 0));    // road cloud

    // Cluster the obstacles to differentiate between various objects
    auto cloudClusters = pointProcessorI.EuclideanClustering(segmentCloud.first, 0.2, 50, 2000);

    std::vector<Color> colors = { Color(0, 1, 1), Color(1, 0.92, 0.016), Color(0, 0, 1) };

    for (const auto& cluster : cloudClusters) {
        renderPointCloud(viewer, cluster, "obstacleCloud" + std::to_string(clusterId), colors[clusterId % colors.size()]);

        Box box = pointProcessorI.BoundingBox(cluster);
        renderBox(viewer, box, clusterId, Color(1, 1, 1), 0.7);

        ++clusterId;
    }
}

/// <summary>
/// setAngle: SWITCH CAMERA ANGLE {XY, TopDown, Side, FPS}
/// </summary>
/// <param name="setAngle"></param>
/// <param name="viewer"></param>
void initCamera(CameraAngle setAngle, pcl::visualization::PCLVisualizer::Ptr& viewer)
{

    viewer->setBackgroundColor (0, 0, 0);
    
    // set camera position and angle
    viewer->initCameraParameters();
    // distance away in meters
    int distance = 16;
    
    switch(setAngle)
    {
        case XY : viewer->setCameraPosition(-distance, -distance, distance, 1, 1, 0); break;
        case TopDown : viewer->setCameraPosition(0, 0, distance, 1, 0, 1); break;
        case Side : viewer->setCameraPosition(0, -distance, 0, 0, 0, 1); break;
        case FPS : viewer->setCameraPosition(-10, 0, 0, 0, 0, 1);
    }

    if(setAngle!=FPS)
        viewer->addCoordinateSystem (1.0);
}


/// <summary>
/// main entry point
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>
int main(int argc, char** argv)
{
    bool showSimpleHighway = false;
    bool showCityBlockSingleFile = false;

    std::cout << "Starting environment" << std::endl;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    initCamera(FPS, viewer);

    if (showSimpleHighway)
    {
        simpleHighway(viewer);
        return 0;
    }

    if (showCityBlockSingleFile)
    {
        cityBlockSingleFile(viewer);
        while (!viewer->wasStopped())
        {
            viewer->spinOnce();
        }
    }
    else
    {
        ProcessPointClouds<pcl::PointXYZI>* pointProcessorI = new ProcessPointClouds<pcl::PointXYZI>();
        std::vector<boost::filesystem::path> stream = pointProcessorI->streamPcd("../../src/sensors/data/pcd/data_1"); // if the build is in : build\Debug\environment.exe
        //std::vector<boost::filesystem::path> stream = pointProcessorI->streamPcd("D:/Udacity/Code/23May/SFND_Lidar_Obstacle_Detection-master/src/sensors/data/pcd/data_2");

        auto streamIterator = stream.begin();

        pcl::PointCloud<pcl::PointXYZI>::Ptr inputCloudI;

        while (!viewer->wasStopped())
        {
            // Clear viewer
            viewer->removeAllPointClouds();
            viewer->removeAllShapes();

            // Load pcd and run obstacle detection process
            inputCloudI = pointProcessorI->loadPcd((*streamIterator).string());
            cityBlockStream(viewer, pointProcessorI, inputCloudI);

            // Iterate to the next pcd file
            streamIterator = (streamIterator == stream.end() - 1) ? stream.begin() : streamIterator + 1;

            viewer->spinOnce();
        }
    }

    return 0;
}
