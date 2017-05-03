#include "MLP.h"
#include "omp.h"

MLP::MLP()
{
    //ctor
}
MLP::MLP(int h1pNo, int h2pNo, double learnDataQtty, int maxIter, bool validationMethod)
{
    //no of neurons in hidden layer
    h1No = h1pNo;
    h2No = h2pNo;
    int testDataQtty = 100 - learnDataQtty;
    int numberOfRestarts = 1, trainingSetSize = 1;
    int subsetLength = 10;
    int subsetSize = -1;
    //param = qtty;
    //this->dL = dL; //learning
    //this->dT = dT; //testing
    //this->decay = wDecay;
    this->maxIter = maxIter;
    this->kFoldValidation = validationMethod;

    //qtt of input features, number of classes to be produced
    alglib::mlpcreatetrainercls(X.getObjectAt(0).getFeatureCount(), X.getClassCount(), trn); 

    double wstep = 0.000;
    //weight  decay  coefficient,  >=0.  Weight  decay  term 'Decay*||Weights||^2' is added to error  function.  If
    //you don't know what Decay to choose, use 1.0E-3. Weight decay can be set to zero,  in this case network
    //is trained without weight decay.
    double wdecay = 0.001;
    // by default we set moderate weight decay
    mlpsetdecay(trn, wdecay); 
    // * we choose iterations limit as stopping condition (another condition - step size - is zero, which means than this condition is not active)
    mlpsetcond(trn, wstep, this->maxIter);     

    if (noOfNeuronsIsMoreThan0())
    {
       //create nn network with noofinput features, 2 hidden layers, noofclasses (and sore to network variable)
        alglib::mlpcreatec2(X.getObjectAt(0).getFeatureCount(), h1No, h2No, X.getClassCount(), network);  
    }
    if (firstIsMoreSecondIsEqualsTo0())
    {
       //create nn network with no of input features, 1 hidden layer, noofclasses (and sore to network variable)
         alglib::mlpcreatec1(X.getObjectAt(0).getFeatureCount(), h1No, X.getClassCount(), network); 
    }
    if (noOfNeuronsIsEqualsTo0InBothLayers())
    {
        //create nn network with no of input features, 0 hidden layer, noofclasses (and sore to network variable)
        alglib::mlpcreatec0(X.getObjectAt(0).getFeatureCount(), X.getClassCount(), network); 
    }
        ///h2No must be non zero

    Initialization();
    //ctor
    // now get network error
    // do not calculate cross-validation since it validates the topology of the network
}

MLP::~MLP()
{
    //dtor
}

ObjectMatrix MLP::getProjection()
{
    //int cols = X.getClassCount();
    int ftCount = X.getObjectAt(0).getFeatureCount();
    int objCount = X.getObjectCount();

    initializeYMatrix(objCount, ftCount + X.getClassCount());

    alglib::real_1d_array tmpYObj;
    alglib::real_1d_array tmpXObj;

    tmpYObj.setlength(ftCount);
    tmpXObj.setlength(X.getClassCount());

    DataObject tmpO;
    
    updateProjectionMatrixY();
    
    std::vector <std::string > probabilities;
    probabilities.reserve(0);

    fillVector();

    Y.addAtributes(probabilities);

    Y.setPrintClass(X.getStringClassAttributes());

    return Y;
}

double MLP::getStress()
{
    if (this->kFoldValidation)
    {
        return rep.avgrelerror;
    }
    else
    {
        return repp.avgrelerror;
    }
//}
    /*
    * Rep.RelCLSError - fraction of misclassified cases.
    * Rep.AvgCE - acerage cross-entropy
    * Rep.RMSError - root-mean-square error
    * Rep.AvgError - average error
    * Rep.AvgRelError - average relative error
    */
  return rep.rmserror;
}

bool MLP::noOfNeuronsIsMoreThan0()
{
    if ((h1No > 0) && (h2No > 0))
    {
        return true;
    }
}

bool MLP::firstIsMoreSecondIsEqualsTo0()
{
   if((h1No > 0) && (h2No == 0))
   {
       return true;
   }
}
bool MLP::noOfNeuronsIsEqualsTo0InBothLayers()
{
    if((h1No == 0) && (h2No == 0))
    {
        return true;
    }
}

int MLP::updateProjectionMatrixY()
{
    for (int i = 0; i < objCount; i++)
    {
        tmpO = X.getObjectAt(i);
        
        updateDataObjectOfY();
        
        alglib::mlpprocess(network, tmpYObj, tmpXObj);

        double max_prob = tmpXObj(0);
        int indx = 0;
        
        updateDataObjectOfYAndSetMaxProbability();

        updateDataObjectClassOfY();
    }
}

int MLP::updateDataObjectOfY()
{
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int ft = 0; ft < ftCount; ft++)
    {
        double feature = tmpO.getFeatureAt(ft);
        tmpYObj(ft) = feature;
        Y.updateDataObject(i, ft, feature);
    }
    return 0;
}

int MLP::updateDataObjectOfYAndSetMaxProbability()
{
    omp_set_num_threads(4);
#pragma omp parallel for    
    for (int j = 0; j < X.getClassCount(); j++)
    {
        Y.updateDataObject(i, j + ftCount, tmpXObj(j));
        if (max_prob < tmpXObj(j))
        {
            max_prob = tmpXObj(j);
            indx = j;
        }
    }
    return 0;
}

int MLP::updateDataObjectClassOfY()
{
    if (tmpO.getClassLabel() != -1)
    {
        Y.updateDataObjectClass(i, tmpO.getClassLabel());
    }
    else
    {
        Y.updateDataObjectClass(i, indx);
    }
       
    return 0;
}

int MLP::fillVector()
{
    omp_set_num_threads(4);
#pragma omp parallel for
    for (int i = 0; i < X.getClassCount(); i++)
    {
        probabilities.push_back("probClass" + X.getStringClassAttributes().at(i));
    }
    
    return 0;
}

int MLP::Initialization()
{
    //do kfold validation
    if (this->kFoldValidation == true) 
    {
        ClusterizationMethods::initializeData();

        //attach learning data to data set
        alglib::mlpsetdataset(trn, ClusterizationMethods::learnSet, ClusterizationMethods::learnObjQtty); 

        alglib::mlpkfoldcv(trn, network, trainingSetSize, int(learnDataQtty), rep);
    }
    else
    {
        ClusterizationMethods::initializeData(learnDataQtty, testDataQtty);

        //attach learning data to data set
        alglib::mlpsetdataset(trn, ClusterizationMethods::learnSet, ClusterizationMethods::learnObjQtty); 

        // train network NRestarts=1, network is trained from random initial state. With NRestarts=0, network is trained without randomization (original state is used as initial point).
        alglib::mlptrainnetwork(trn, network, numberOfRestarts, rep); 
        alglib::integer_1d_array Subset;
        Subset.setlength(subsetLength);
        alglib::mlpallerrorssubset(network, testSet, testObjQtty, Subset, subsetSize, repp);
    }
    
    return 0;
}
    
    
