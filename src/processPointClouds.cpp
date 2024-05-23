// PCL lib Functions for processing point clouds 

#include "processPointClouds.h"


//constructor:
template<typename PointT>
ProcessPointClouds<PointT>::ProcessPointClouds() {}


//de-constructor:
template<typename PointT>
ProcessPointClouds<PointT>::~ProcessPointClouds() {}


template<typename PointT>
void ProcessPointClouds<PointT>::numPoints(typename pcl::PointCloud<PointT>::Ptr cloud)
{
    std::cout << cloud->points.size() << std::endl;
}

/// <summary>
/// FilterCloud applies voxel grid filtering to a point cloud, reducing the number of points by cropping a region for processing. 
/// This streamlined approach accelerates clustering, as fewer points need to be processed compared to the entire scene.
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cloud"></param>
/// <param name="filterRes"></param>
/// <param name="minPoint"></param>
/// <param name="maxPoint"></param>
/// <returns></returns>
template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::FilterCloud(typename pcl::PointCloud<PointT>::Ptr cloud, float filterRes, Eigen::Vector4f minPoint, Eigen::Vector4f maxPoint)
{

    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // Voxel grid point reduction
    typename pcl::PointCloud<PointT>::Ptr cloudFiltered{ new pcl::PointCloud<PointT> };
    pcl::VoxelGrid<PointT> voxelGridFilter;
    voxelGridFilter.setInputCloud(cloud);
    voxelGridFilter.setLeafSize(filterRes, filterRes, filterRes);
    voxelGridFilter.filter(*cloudFiltered);

    // use crop box to select the region of interest and keep these points for processing
    typename pcl::PointCloud<PointT>::Ptr cloudRegion{ new pcl::PointCloud<PointT> }; // the resultant cloud data after applying a crop box (region of interest)
    pcl::CropBox<PointT> cropBoxFilter(true);   // True = remove these indices from the resultant cloud data
    cropBoxFilter.setMin(minPoint);
    cropBoxFilter.setMax(maxPoint);
    cropBoxFilter.setInputCloud(cloudFiltered);
    cropBoxFilter.filter(*cloudRegion);

    // Get the indices of rooftop points
    std::vector<int> roofIndices;   // holds the ego car roof points to be removed later
    pcl::CropBox<PointT> roofFilter(true);    // CropBox (bool extract_removed_indices=false). True = remove these indices from the resultant cloud data
    const Eigen::Vector4f minRoof(-1.5, -1.7, -1, 1);   // the renderBox() function from enviornment.cpp was used to estimate the roof bounding box
    const Eigen::Vector4f maxRoof(2.6, 1.7, 0.4, 1);
    roofFilter.setMin(minRoof);
    roofFilter.setMax(maxRoof);
    roofFilter.setInputCloud(cloudRegion);
    roofFilter.filter(roofIndices);

    // add the indices to PointIndices pointer reference to be used when removing roof points out of the PCD
    pcl::PointIndices::Ptr inliers{ new pcl::PointIndices };
    for (int point : roofIndices)
        inliers->indices.push_back(point);  // indicies object built-in into inliers PointIndicies object

    // seperate the roof indices from the point cloud
    pcl::ExtractIndices<PointT> extract;
    extract.setIndices(inliers);
    extract.setNegative(true);  // remove the indices points
    extract.setInputCloud(cloudRegion);
    extract.filter(*cloudRegion);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "downsampled original " << cloud->points.size() << " points to " << cloudRegion->points.size() << std::endl;
    std::cout << "filtering cloud took " << elapsedTime.count() << " milliseconds" << std::endl;

    return cloudRegion;

}

/// <summary>
///  Seperates inlier points (road points) from the overall cloud data. Returns the road cloud and obstacle cloud.
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="inliers"></param>
/// <param name="cloud"></param>
/// <returns></returns>
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SeparateClouds(pcl::PointIndices::Ptr inliers, typename pcl::PointCloud<PointT>::Ptr cloud) 
{
    // TODO: Create two new point clouds, one cloud with obstacles and other with segmented plane
    typename pcl::PointCloud<PointT>::Ptr obstacleCloud(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr planeCloud(new pcl::PointCloud<PointT>());

    // iterate through the inliers (road plane) and add the associate cloud points to the planeCloud object. 
    // the planeCloud object will be used to seperate the overall point cloud with road and wanted obstacles (cars, trees)
    // use -> because inliers is a pointer object
    for (int inlierIdx : inliers->indices)
    {
        planeCloud->points.push_back(cloud->points[inlierIdx]);
    }

    // Create the filtering object
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);   // the entire point cloud data
    extract.setIndices(inliers);    // used to seperate plane - road points
    extract.setNegative(true);  // [in]	negative	false = normal filter behavior (default), true = inverted behavior.
    extract.filter(*obstacleCloud);   // seperates the cloud data into obstacles and the road plane. All the points that are not inliers are kept here

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult(obstacleCloud, planeCloud);
    return segResult;
}


/// <summary>
/// Segment plane using pcl built in functions
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cloud"></param>
/// <param name="maxIterations"></param>
/// <param name="distanceThreshold"></param>
/// <returns></returns>
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlane(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Time segmentation process
    auto startTime = std::chrono::steady_clock::now();

    // TODO:: Fill in this function to find inliers for the cloud.
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients()); // define coefficients for the model - used to define plane and can be rendered in PCL viewer
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());    // used to seperate point cloud into 2 pieces
    // Create the segmentation object
    pcl::SACSegmentation<PointT> seg;

    // Optional
    seg.setOptimizeCoefficients(true);      // try to get best model
    // Mandatory
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC); // random sample concensus
    seg.setMaxIterations(maxIterations);
    seg.setDistanceThreshold(distanceThreshold);

    // segment the largest planar component from the input cloud
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);    // coefficients can be used to render plane. inliars used to seperate plane (road points)

    if (inliers->indices.size() == 0)   // didn't find any model that can fit this data
    {
        std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
    }

    std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> segResult = SeparateClouds(inliers, cloud);

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "plane segmentation took " << elapsedTime.count() << " milliseconds" << std::endl;

    return segResult;
}

#pragma region Ransac

/// <summary>
/// Segment with ransac algorithm. spliting the cloud to road plane and object planes.
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cloud"></param>
/// <param name="maxIterations"></param>
/// <param name="distanceThreshold"></param>
/// <returns></returns>
template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr, typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::SegmentPlaneWithRansac(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceThreshold)
{
    // Run RANSAC to find inliers for the plane
    std::unordered_set<int> inliers = Ransac3DHelper(cloud, maxIterations, distanceThreshold);

    // Initialize PointIndices with inliers
    pcl::PointIndices::Ptr inlierIndices(new pcl::PointIndices());
    inlierIndices->indices.assign(inliers.begin(), inliers.end());

    // Separate the cloud into two parts: inliers (plane) and outliers (obstacles)
    auto segmentedCloud = SeparateClouds(inlierIndices, cloud);

    return segmentedCloud;
}

/// <summary>
/// separate cloud
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cloud"></param>
/// <param name="maxIterations"></param>
/// <param name="distanceTol"></param>
/// <returns></returns>
template<typename PointT>
std::unordered_set<int> ProcessPointClouds<PointT>::Ransac3DHelper(typename pcl::PointCloud<PointT>::Ptr cloud, int maxIterations, float distanceTol)
{
    auto startTime = std::chrono::steady_clock::now();
    std::unordered_set<int> inliersResult;
    srand(time(NULL));

    // For max iterations
    for (int iteration = 0; iteration < maxIterations; ++iteration)
    {
        // Randomly pick 3 points to create a plane
        std::unordered_set<int> inliers;
        while (inliers.size() < 3)
            inliers.insert(rand() % cloud->points.size());

        auto itr = inliers.begin();
        const auto& p1 = cloud->points[*itr++];
        const auto& p2 = cloud->points[*itr++];
        const auto& p3 = cloud->points[*itr];

        // Compute plane coefficients A, B, C, D
        float a = (p2.y - p1.y) * (p3.z - p1.z) - (p2.z - p1.z) * (p3.y - p1.y);
        float b = (p2.z - p1.z) * (p3.x - p1.x) - (p2.x - p1.x) * (p3.z - p1.z);
        float c = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
        float d = -(a * p1.x + b * p1.y + c * p1.z);

        // Measure distance between every point and the fitted plane
        for (int index = 0; index < cloud->points.size(); ++index)
        {
            if (inliers.count(index) > 0)
                continue;

            const auto& point = cloud->points[index];
            float distance = fabs(a * point.x + b * point.y + c * point.z + d) / sqrt(a * a + b * b + c * c);

            if (distance <= distanceTol)
                inliers.insert(index);
        }

        // Update the best inliers set
        if (inliers.size() > inliersResult.size())
            inliersResult = inliers;
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Ransac3D took " << elapsedTime.count() << " milliseconds." << std::endl;

    return inliersResult;
}

#pragma endregion Ransac



/// <summary>
/// Clustering using pcl inbuilt functions
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cloud"></param>
/// <param name="clusterTolerance"></param>
/// <param name="minSize"></param>
/// <param name="maxSize"></param>
/// <returns></returns>
template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::Clustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize)
{
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;

    // TODO:: Fill in the function to perform euclidean clustering to group detected obstacles
    // Creating the KdTree object for the search method of the extraction
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(clusterTolerance); // 2cm
    ec.setMinClusterSize(minSize);
    ec.setMaxClusterSize(maxSize);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    for (pcl::PointIndices getIndices : cluster_indices) {
        typename pcl::PointCloud<PointT>::Ptr cloud_cluster(new pcl::PointCloud<PointT>);  // holds obstacles in the point cloud
        for (int index : getIndices.indices) {
            cloud_cluster->points.push_back(cloud->points[index]);
        }
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        clusters.push_back(cloud_cluster);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


#pragma region EuclideanClustering

/// <summary>
///  luster points via Euclidean Clustering algorithm
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cloud"></param>
/// <param name="clusterTolerance"></param>
/// <param name="minSize"></param>
/// <param name="maxSize"></param>
/// <returns></returns>
template<typename PointT>
std::vector<typename pcl::PointCloud<PointT>::Ptr> ProcessPointClouds<PointT>::EuclideanClustering(typename pcl::PointCloud<PointT>::Ptr cloud, float clusterTolerance, int minSize, int maxSize) {
    // Time clustering process
    auto startTime = std::chrono::steady_clock::now();

    std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters; // stores the found clusters into a point cloud object
    std::vector<std::vector<float>> cloudPoints;    // stores the cloud points as input for the euclideanCluster() function

    // Initialize KD-Tree
    KdTree* tree = new KdTree;

    // Insert points into the KD-Tree and prepare point vector
    for (const auto& point : cloud->points) {
        std::vector<float> pointVector = { point.x, point.y, point.z };
        tree->insert(pointVector, &point - &cloud->points[0]);
        cloudPoints.push_back(pointVector);
    }

    // Perform Euclidean clustering
    std::vector<std::vector<int>> euclideanClusters = euclideanCluster(cloudPoints, tree, clusterTolerance);

    // Add the clusters to the point cloud
    for (const auto& cluster : euclideanClusters) {
        // Check if the points within the cluster satisfy the min and max size specified from the function arguments
        if (cluster.size() >= minSize && cluster.size() < maxSize) {
            typename pcl::PointCloud<PointT>::Ptr cloudCluster(new pcl::PointCloud<PointT>); // holds obstacles in the point cloud
            for (int index : cluster) {
                cloudCluster->points.push_back(cloud->points[index]);
            }
            cloudCluster->width = cloudCluster->points.size();
            cloudCluster->height = 1;
            cloudCluster->is_dense = true;

            clusters.push_back(cloudCluster);
        }
    }

    // Clean up KD-Tree
    delete tree;

    // Measure elapsed time
    auto endTime = std::chrono::steady_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    std::cout << "Euclidean clustering took " << elapsedTime.count() << " milliseconds and found " << clusters.size() << " clusters" << std::endl;

    return clusters;
}


/// <summary>
/// clusterHelper
/// </summary>
/// <param name="index"></param>
/// <param name="points"></param>
/// <param name="cluster"></param>
/// <param name="processed"></param>
/// <param name="tree"></param>
/// <param name="distanceTol"></param>
template<typename PointT>
void ProcessPointClouds<PointT>::clusterHelper(int index, const std::vector<std::vector<float>>& points, std::vector<int>& cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol)
{
    std::queue<int> q;
    q.push(index);

    while (!q.empty()) {
        int currentIndex = q.front();
        q.pop();

        if (!processed[currentIndex]) {
            processed[currentIndex] = true;
            cluster.push_back(currentIndex);

            std::vector<int> nearbyPoints = tree->search(points[currentIndex], distanceTol);
            for (int id : nearbyPoints) {
                if (!processed[id]) {
                    q.push(id);
                }
            }
        }
    }
}

/// <summary>
/// euclideanCluster
/// </summary>
/// <param name="points"></param>
/// <param name="tree"></param>
/// <param name="distanceTol"></param>
/// <returns></returns>
template<typename PointT>
std::vector<std::vector<int>> ProcessPointClouds<PointT>::euclideanCluster(const std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol)
{
    std::vector<std::vector<int>> clusters;
    std::vector<bool> processed(points.size(), false);

    for (int i = 0; i < points.size(); ++i) {
        if (!processed[i]) {
            std::vector<int> cluster;
            clusterHelper(i, points, cluster, processed, tree, distanceTol);
            clusters.push_back(cluster);
        }
    }

    return clusters;
}

#pragma endregion EuclideanClustering

template<typename PointT>
Box ProcessPointClouds<PointT>::BoundingBox(typename pcl::PointCloud<PointT>::Ptr cluster)
{

    // Find bounding box for one of the clusters
    PointT minPoint, maxPoint;
    pcl::getMinMax3D(*cluster, minPoint, maxPoint);

    Box box;
    box.x_min = minPoint.x;
    box.y_min = minPoint.y;
    box.z_min = minPoint.z;
    box.x_max = maxPoint.x;
    box.y_max = maxPoint.y;
    box.z_max = maxPoint.z;

    return box;
}

/// <summary>
/// Create bounding box which fits better
/// </summary>
/// <typeparam name="PointT"></typeparam>
/// <param name="cluster"></param>
/// <returns></returns>
template<typename PointT>
BoxQ ProcessPointClouds<PointT>::BoundingBoxQ(typename pcl::PointCloud<PointT>::Ptr cluster)
{
    // Create an object to compute the moment of inertia and the oriented bounding box
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(cluster);
    feature_extractor.compute();

    // Define variables to store the results
    PointT min_point_OBB, max_point_OBB;
    PointT position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;

    // Get the oriented bounding box
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    // Convert rotational matrix to quaternion
    Eigen::Quaternionf bboxQuaternion(rotational_matrix_OBB);

    // Calculate cube dimensions
    float cube_length = max_point_OBB.x - min_point_OBB.x;
    float cube_width = max_point_OBB.y - min_point_OBB.y;
    float cube_height = max_point_OBB.z - min_point_OBB.z;

    // Fill the BoxQ structure
    BoxQ boxQ;
    boxQ.bboxTransform = Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);
    boxQ.bboxQuaternion = bboxQuaternion;
    boxQ.cube_length = cube_length;
    boxQ.cube_width = cube_width;
    boxQ.cube_height = cube_height;

    return boxQ;
}


template<typename PointT>
void ProcessPointClouds<PointT>::savePcd(typename pcl::PointCloud<PointT>::Ptr cloud, std::string file)
{
    pcl::io::savePCDFileASCII (file, *cloud);
    std::cerr << "Saved " << cloud->points.size () << " data points to "+file << std::endl;
}


template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr ProcessPointClouds<PointT>::loadPcd(std::string file)
{

    typename pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);

    if (pcl::io::loadPCDFile<PointT> (file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file \n");
    }
    std::cerr << "Loaded " << cloud->points.size () << " data points from "+file << std::endl;

    return cloud;
}


template<typename PointT>
std::vector<boost::filesystem::path> ProcessPointClouds<PointT>::streamPcd(std::string dataPath)
{

    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{dataPath}, boost::filesystem::directory_iterator{});

    // sort files in accending order so playback is chronological
    sort(paths.begin(), paths.end());

    return paths;

}