#ifndef __UNSCENTED_PARTICLE_FILTER_HPP__
#define __UNSCENTED_PARTICLE_FILTER_HPP__

#include "../PFV4/PF.hpp"
#include <opencv2/core.hpp>

class UPF : public PF
{
private:
/********UNSCENTED KALMAN FILTER STATISTICS****************************/	
	cv::Mat mS;
	cv::Mat mKGain = cv::Mat(6, 3, CV_32F);
	//AUXILIAR MATRICES TO HOLD PARTICLES COVARIANCE
	cv::Mat mDS1, mDS2;
	//POINTER TO NEXT PARTICLES BUFFER
	cv::Mat *mSNextBuffer; 
	
	//SUT PARAMETERS
	float mAlpha = 0.5;
	float mBeta = 2;
	float mKapa = 0;
	float mn = 6;
	//COMPUTE LAMBDA
	float mLambda = mAlpha*mAlpha*(mn + mKapa) - mn; 

	/**ALLOCATE SPACE FOR SIGMA POINTS WEIGHTS AND INITIALIZE IT*/
	cv::Mat mWm = cv::Mat(2*mn + 1, 1, CV_32F);
	cv::Mat mWc = cv::Mat(2*mn + 1, 1, CV_32F);
	
	cv::Mat mChi = cv::Mat(mn, 2*mn + 1, CV_32F);
	cv::Mat mChiPlus = mChi(cv::Rect(1, 0, mn, mn));
	cv::Mat mChiMinus = mChi(cv::Rect(mn + 1, 0, mn, mn));
	cv::Mat mChiP, mChiV;
	cv::Mat mGamma = mChi(cv::Rect(0, 0, 2*mn + 1, 3));
	
	cv::Mat mSa = cv::Mat(mn, mn, CV_32F); //AUXILIAR COVARIANCE MATRIX
	cv::Mat mL = cv::Mat(mn, mn, CV_32F);
	
	cv::Mat mZ = cv::Mat(3, 1, CV_32F);
	cv::Mat mSz = cv::Mat(3, 3, CV_32F);
	cv::Mat mSxz = cv::Mat(mn, 3, CV_32F);

	cv::Mat mXPred = cv::Mat(6, 1, CV_32F);
	cv::Mat mXMean = cv::Mat(6, 1, CV_32F);
	cv::Mat mSPred = cv::Mat(6, 6, CV_32F);

	/**PRIVATE METHODS*/
	void computeSigmaPts(uint);

public:
/*******CONSTRUCTORS & DESTRUCTORS*************************************/
	UPF(std::tuple<float, float, float, cv::Mat, cv::Mat, cv::Mat, uint, 
		float, cv::Mat, cv::Mat, cv::Mat, cv::Mat, float, float, uint>);
	//~UPF();
	
/*******METHODS********************************************************/
	void predictUpdate(cv::Mat &, cv::Mat);
	void choleskyDecomp(cv::Mat &, cv::Mat &);
	void sysResampling() override;
	float computeHistSims(cv::Mat &) override;
};

#endif //__UNSCENTED_PARTICLE_FILTER_HPP__
