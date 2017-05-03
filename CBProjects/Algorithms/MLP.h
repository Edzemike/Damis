#ifndef MLP_H
#define MLP_H

#include "ClusterizationMethods.h"

/*! \file MLP class
    \brief A class of methods and attributes for multilayer perceptron algorithm.
 */

class MLP : public ClusterizationMethods
{
    public:
        /*! \fn MLP();
        * \brief A default constructor.
        */
        MLP();
        /*! \fn MLP(int h1pNo, int h2pNo, double qtty, int maxIter, bool validationMethod);
        * \brief An overloaded constructor.
        * \param  double h1pNo - number of neurons in first layer.
        * \param double h2No - number of neurons in second layer
        * \param double qtty - either kfold validation or percentage for testing
        * \param in maxIter - number of mafimum iteration
        * \param  bool validationMEthod - kfold or corss validation.
        */
        MLP(int h1pNo, int h2pNo, double qtty, int maxIter, bool validationMethod);
        /*! \fn virtual ~MLP();
        * \brief A destructor.
        */
        virtual ~MLP();

        /*! \fn virtual ObjectMatrix getProjection();
         *  \brief Returns the projection matrix \a Y of matrix \a X.
         *  \return ObjectMatrix Y - the projection matrix.
         */
        virtual ObjectMatrix getProjection();
        /*! \fn double getStress();
         *  \brief Returns the projection matrix \a Y of matrix \a X.
         *  \return double  - average relative error.
         */
        double getStress();
        
        /*! \fn bool noOfNeuronsIsMoreThan0();
         *  \brief Returns true if number of neurons in both layers are more than 0.
         *  \return bool - true;
         */
        bool noOfNeuronsIsMoreThan0();
    
        /*! \fn bool firstIsMoreSecondIsEqualsTo0()
         *  \brief Returns true if number of neurons in first layer is more than 0 and equals to zero in second layer.
         *  \return bool - true;
         */
        bool firstIsMoreSecondIsEqualsTo0();
        
        /*! \fn bool noOfNeuronsIsEqualsTo0InBothLayers()
         *  \brief Returns true if number of neurons in both layers is equals to zero.
         *  \return bool - true;
         */
        bool noOfNeuronsIsEqualsTo0InBothLayers();
        
        /*! \fn int updateProjectionMatrixY()
         *  \brief Updates projection matrix Y.
         *  \return int 0;
         */
        int updateProjectionMatrixY();
        
        /*! \fn int updateDataObjectOfY()
         *  \brief Updates DataObject property of Y.
         *  \return int 0;
         */
        int updateDataObjectOfY();
    
        /*! \fn int updateDataObjectOfYAndSetMaxProbability()
         *  \brief Updates DataObject property of Y and sets maximum probability.
         *  \return int 0;
         */
        int updateDataObjectOfYAndSetMaxProbability();
    
        /*! \fn int updateDataObjectClassOfY()
         *  \brief Updates DataObjectClass property of Y.
         *  \return int 0;
         */
        int updateDataObjectClassOfY();
    
        /*! \fn int fillVector()
         *  \brief Fills vector with values.
         *  \return int 0;
         */
        int fillVector();
    
        /*! \fn int Initialization()
         *  \brief Attaches learning data to data set and trains network.
         *  \return int 0;
         */
        int Initialization();
    
    private:

        /*! \var int h1No;
         *  \brief Number of neurons in first hidden layer
         */
        int h1No;
        /*! \var int h2No;
         *  \brief Number of neurons in first hidden layer
         */
        int h2No;
        /*! \var int maxIter;
         *  \brief Number of iterations
         */
        int maxIter;
        /*! \var  bool kFoldValidation;
         *  \brief Indicates if perform k-fold validation
         */
        bool kFoldValidation;
        /*! \var alglib::mlptrainer trn;
         *  \brief Algrlib structure used of MLP training
         */
        alglib::mlptrainer trn;
        /*! \var alglib::multilayerperceptron network;
        *  \brief Alglib structure that describe the network
        */
        alglib::multilayerperceptron network;
        /*! \var alglib::mlpreport rep;
        *  \brief Algrlib structure that holds MLP reports
        */
        alglib::mlpreport rep;
        /*! \var alglib::modelerrors repp;
        *  \brief Algrlib structure that holds network model reports
        */
        alglib::modelerrors repp;

};

#endif // MLP_H
