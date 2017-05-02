#include "KMEANS.h"
#include "alglib/dataanalysis.h"
#include "AdditionalMethods.h"
#include <iostream>

KMEANS::KMEANS()
{
    //ctor

}
KMEANS::~KMEANS()
{
    //dtor
}

KMEANS::KMEANS(int noOfClusters, int maxIter):ClusterizationMethods(noOfClusters)
{
    this->maxIter = maxIter;
    ClusterizationMethods::setYMatrixDimensions(X.getObjectCount(), X.getObjectAt(0).getFeatureCount()); // set row feature count to 2
    ClusterizationMethods::initializeYMatrix(); //fill matrix with random data
}

ObjectMatrix KMEANS::getProjection()
{
    alglib::clusterizerstate clusterizerState = {};
    alglib::KmeansReport kmeansReport = {};
    alglib::real_2d_array inputArray = {};
    int rowsX = X.getObjectCount(), colsX = X.getObjectAt(0).getFeatureCount();
    inputArray.setlength(rowsX, colsX);

    for ( int i = 0; i < rowsX; i++ ) // convert X matrix to alglib 2d array of reals
    {
        DataObject tmp = X.getObjectAt(i);
        for ( int j = 0; j < colsX; j++ )
        {
            inputArray(i,j) = tmp.getFeatureAt(j);
        }
    }

    alglib::clusterizercreate(clusterizerState);
    alglib::clusterizersetpoints(clusterizerState, inputArray, 2); //2 means Euclidean distances
    alglib::clusterizersetkmeanslimits(clusterizerState, 1, this->maxIter);        //1 means quantity of restarts, if maxIter = 0 then unlimited number of iterations are performed
    alglib::clusterizerrunkmeans(clusterizerState, ClusterizationMethods::getNoOfClusters(), kmeansReport);

    Y = X;
    std::vector <std::string> classes; classes.reserve(0);
    for (int i = 0; i < ClusterizationMethods::getNoOfClusters(); i++)
        classes.push_back(std::to_string(static_cast<long long>(i)));
    Y.setPrintClass(classes);

    if (int(kmeansReport.terminationtype) == 1) //if success
    {
        for ( int i = 0; i < ClusterizationMethods::getNoOfReturnRows(); i++ ) // updates return Y matrix class values (sets to those returned by kmeans)
                Y.updateDataObjectClass(i, kmeansReport.cidx(i));
    }
     return Y;
}

double KMEANS::getStress()
{
    return 0.0;
}
