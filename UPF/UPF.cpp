#include "UPF.hpp"

#include <fstream>

#define PI	3.1415926535897932384626433832795
#define WKte 61528.90838881947 //(2*pi)^6

UPF::UPF(std::tuple<float, float, float, cv::Mat, cv::Mat, cv::Mat, uint, 
	float, cv::Mat, cv::Mat, cv::Mat, cv::Mat, float, float, uint> data) 
		: PF(data)
{ 
	/**UNSCENTED KALMAN FILTER VARIABLES*/
	//ALLOCATE BUFFERS TO STORE COVARIANCES
	mDS1 = cv::Mat(6, 6*mnParticles, CV_32F);
	mDS2 = cv::Mat(6, 6*mnParticles, CV_32F);

	//POINT TO mDP1 INITIALLY
	mS = mDS1(cv::Rect(0, 0, 6*mnParticles, 6));
	mSNextBuffer = &mDS2;	

	//SET PARTICLE COVARIANCES
	cv::Mat S;
	for(uint i = 0; i < mnParticles; i++)
		S = mS(cv::Rect(6*i, 0, 6, 6)), mInitVar.copyTo(S);
	
	//COMPUTE SUT WEIGHTS
	mWm = 0.5/(mLambda + mn);
	mWc = 0.5/(mLambda + mn);
	*mWm.ptr<float>(0) = mLambda/(mLambda + mn);
	*mWc.ptr<float>(0) = mLambda/(mLambda + mn) + 1 - mAlpha*mAlpha 
																+ mBeta;

	mChiP = mChi.rowRange(0, 3), mChiV = mChi.rowRange(3, 6);

	mXa = cv::Mat(mChi.size(), CV_32F);
	mPa = mXa.rowRange(0, 3);
	
	mQ = cv::Mat(3, 2*mn + 1, CV_32F);

	mQ.row(0) = 0.0f;
	mQ.row(1) = mMotionModel ? 9810.0f : 0.0f;
	mQ.row(2) = 0.0f;

	/*mObsVar = cv::Mat(6, 6, CV_32F);
	
	mObsVar.at<float>(0, 0) = 25.0f;
	mObsVar.at<float>(1,1) = 25.0f;
	mObsVar.at<float>(2,2) = 25.0f;
	mObsVar.at<float>(3,3) = 25.0f;
	mObsVar.at<float>(4,4) = 25.0f;
	mObsVar.at<float>(5,5) = 25.0f;*/
}


void UPF::predictUpdate(cv::Mat &im, cv::Mat z)
{
	//UPDATE RANDOM NUMBER GENERATOR SEED
	cv::theRNG().state = cv::getTickCount();

	cv::Mat b = cv::Mat(6, 1, CV_32F), L = cv::Mat(6, 6, CV_32F);
	
	auto e1 = cv::getTickCount();
	for(uint i = 0; i < mnParticles; i++)
	{
		computeSigmaPts(i);
		
		if(!mE)
			mChi = mA*mChi + mB*mQ;
		else
		{
			float t, t1, t2, a, b, c, d;
			mXa = mA*mChi + mB*mQ; //AUXILIAR MOTION MODEL
			//VECTOR OF DISTANCES TO OBSTACLE
			mDa = mNt*(mPa - mD0.colRange(0, 2*mn + 1)); 
			for(uint k = 0; k < 2*mn + 1; k++)
			{				
				if(*mDa.ptr<float>(0, k) > 0.0f)
					mXa.col(k).copyTo(mChi.col(k));
				//IF COLLISON HAPPENED, GET time of collision
				else if((mMotionModel)) 
				{
					//COMPUTE time of collision
					a = *((cv::Mat)(mNt*mQ.col(k))).ptr<float>(0);
					b = *((cv::Mat)(mNt*mChiV.col(k))).ptr<float>(0);
					c = *((cv::Mat)(mNt*(mChiP.col(k) 
									- mD0.col(0)))).ptr<float>(0);
					d = b*b - 2.0f*a*c;

					//IF TIME DOES NOT MEET RESTRICTIONS, DO NOTHING
					if(d < 0.0f)
						continue;
					else
					{
						t1 = (-b + sqrt(d))/a, t2 = (-b - sqrt(d))/a;
						if(t1 > 0.0f && t1 < mDT)
							t = t1;
						else if(t2 > 0.0f && t2 < mDT)
							t = t2;
						else
							continue;
					}				
					//ITERATE TILL TIME = TIME OF COLLISION	
					mChiP.col(k) = mChiP.col(k) + mChiV.col(k)*t + 
								   mQ.col(k)*0.5f*t*t;
					mChiV.col(k) = mChiV.col(k) + mQ.col(k)*t;
					
					//APPLY IMPACT MODEL
					mChiV.col(k) = mChiV.col(k) - *((cv::Mat)
						(mNt*mChiV.col(k))).ptr<float>(0)*(1.0f + mE)*mN;
					
					//ITERATE TILL TIME = SAMPLING TIME
					mChiP.col(k) = mChiP.col(k) + mChiV.col(k)*(mDT - t) 
						+ mQ.col(k)*0.5f*(mDT - t)*(mDT - t);
					mChiV.col(k) = mChiV.col(k) + mQ.col(k)*(mDT - t);
				}
				else
				{
					//COMPUTE time of collision
					b = *((cv::Mat)(mNt*mChiV.col(k))).ptr<float>(0);
					c = *((cv::Mat)(mNt*(mChiP.col(k) 
									- mD0.col(0)))).ptr<float>(0);

					t = -c/b;
					if(t < 0.0f)
						continue;

					//ITERATE TILL TIME = TIME OF COLLISION	
					mChiP.col(k) = mChiP.col(k) + mChiV.col(k)*t;
					
					//APPLY IMPACT MODEL
					mChiV.col(k) = mChiV.col(k) - *((cv::Mat)
						(mNt*mChiV.col(k))).ptr<float>(0)*(1.0f + mE)*mN;
					
					//ITERATE TILL TIME = SAMPLING TIME
					mChiP.col(k) = mChiP.col(k) + mChiV.col(k)*(mDT - t);				
				}
			}
		}
	
		mX.col(i) = mChi*mWm;

		mZ = mX(cv::Rect(i, 0, 1, 3)); //mGamma*mWm;

		cv::Mat r1;
		cv::Mat S = mS(cv::Rect(6*i, 0, 6, 6)); S = 0.0f;
		for(uint j = 0; j < 2*mn + 1; j++)
		{
			r1 = mChi.col(j) - mX.col(i);
			S += (*mWc.ptr<float>(j))*(r1*r1.t());
		}
		S(cv::Rect(0, 0, mSz.cols, mSz.rows)).copyTo(mSz);
		S(cv::Rect(0, 0, mSxz.cols, mSxz.rows)).copyTo(mSxz);
		
		S += mModelVar, mSz += mObsVar;
		
		//TO BE USED IN WEIGHT CALCULATION
		mX.col(i).copyTo(mXPred), S.copyTo(mSPred);

		//CALCULATE KALMAN GAIN
		mKGain = mSxz*mSz.inv();

		//INSERT MEASUREMENT
		mX.col(i) += mKGain*(z - mZ); 
		
		//TO BE USED IN WEIGHT CALCULATION
		mX.col(i).copyTo(mXMean);
		
		//UPDATE COVARIANCE MATRIX
		S -= mKGain*mSz*mKGain.t();

		/**SAMPLE x ~ N(x, P)*/
		cv::randn(b, 0.0f, 1.0f); //MAKE N(0, 1)
		choleskyDecomp(S, L); //COMPUTE CHOLESKY DECOMPOSITION
				
		mX.col(i) += L*b;

		cv::Mat rPred = mX.col(i) - mXPred, rMean = mX.col(i) - mXMean;		

		float pred = exp(*((cv::Mat)(-.5f*rPred.t()*mSPred.inv()*rPred))
				.ptr<float>(0))/sqrt(WKte*cv::determinant(mSPred));
		float prop = exp(*((cv::Mat)(-.5f*rMean.t()*mSPred.inv()*rMean))
				.ptr<float>(0))/sqrt(WKte*cv::determinant(S));
			
		*mW.ptr<float>(0, i) = prop;
	}

	//OBSERVATION STAGE
	auto e2 = cv::getTickCount();
	//std::cout << " " << (e2 - e1)/cv::getTickFrequency() << " ";

	e1 = cv::getTickCount();
	projectParticles();
	e2 = cv::getTickCount();
	//std::cout << " Proj " << (e2 - e1)/cv::getTickFrequency() << " ";
	
	e1 = cv::getTickCount();
	//COMPUTE HISTOGRAMS AND WEIGHTS AND NORMALIZE WEIGHTS
	mW /= computeHistSims(im);
	e2 = cv::getTickCount();
	//std::cout << " Hist " << (e2 - e1)/cv::getTickFrequency() << "\n";
}

void UPF::computeSigmaPts(uint i)
{
	//PUT X IN SIGMA POINTS
	mChi.row(0) = *mX.ptr<float>(0, i);
	mChi.row(1) = *mX.ptr<float>(1, i);
	mChi.row(2) = *mX.ptr<float>(2, i);
	mChi.row(3) = *mX.ptr<float>(3, i);
	mChi.row(4) = *mX.ptr<float>(4, i);
	mChi.row(5) = *mX.ptr<float>(5, i);

	//MAKE (n + lambda)*P
	mS.colRange(6*i, 6*(i + 1)).copyTo(mSa); mSa *= mn + mLambda;
	
	//COMPUTE SQUARE ROOT AND ADD COVARIANCE TO SIGMA POINTS
	choleskyDecomp(mSa, mL), mChiPlus += mL, mChiMinus += -mL;
}
 
void UPF::sysResampling()
{
	uint j = 0;
	float u = cv::randu<float>()/mnParticles;
	float sum = *mW.ptr<float>(j);
	cv::Mat S, Starget;

	for(uint i = 0; i < mnParticles; i++)
	{
		while(sum < u && j < mnParticles - 1)
			sum += *mW.ptr<float>(0, ++j);
		mX.col(j).copyTo(mNextBuffer->col(i));
		S = mS(cv::Rect(6*j, 0, 6, 6));
		Starget = (*mSNextBuffer)(cv::Rect(6*i, 0, 6, 6));
		S.copyTo(Starget);
		u += 1.0/mnParticles;
	}
	
	//SWAP POINTERS TO NEXT BUFFER
	mX = (*mNextBuffer).rowRange(0, 6);
	mP = mX.rowRange(0, 3), mV = mX.rowRange(3, 6);
	mNextBuffer == &mD2 ? mNextBuffer = &mD1, mSNextBuffer = &mDS1 : 
						  mNextBuffer = &mD2, mSNextBuffer = &mDS2;
}

void UPF::choleskyDecomp(cv::Mat &A, cv::Mat &L)
{
	//MAKE CHOLESKY DECOMPOSITON STABLE
	A += cv::Mat::eye(A.cols, A.cols, CV_32F)*0.00001;
	
	L = 0.0f; 
	float s;
	for(uint k = 0; k < (uint)A.cols; k++)
	{	
		for(uint i = 0; i < k; i++)
		{
			s = 0.0f;	
			for(uint j = 0; j < i; j++)			
				s += *L.ptr<float>(i, j)*(*L.ptr<float>(k, j));
			*L.ptr<float>(k, i) = 
				(*A.ptr<float>(k, i) - s)/(*L.ptr<float>(i, i));
		}
		s = 0.0f;
		for(uint j = 0; j < k; j++)
			s += *L.ptr<float>(k, j)*(*L.ptr<float>(k, j));
		*L.ptr<float>(k, k) = sqrt(*A.ptr<float>(k, k) - s);	
	}
}

float UPF::computeHistSims(cv::Mat &im)
{	
	float sum_lk = 0.0f;
	
	for(uint i = 0; i < mnParticles; i++)
	{
		//RESET HISTOGRAMS
		mHists = 0.0f; mUsedPts[0] = mUsedPts[1] = 0; 

		uint k, xp, yp; cv::Vec3b bgr;

		for(k = 0; k < mnPts; k++)
		{
			xp = *mPx.ptr<float>(0, i*mnPts + k);
			yp = *mPx.ptr<float>(1, i*mnPts + k);

			if(xp > (uint)im.cols || yp > (uint)im.rows)
				continue;
			bgr = im.at<cv::Vec3b>(cv::Point(xp, yp));
			//INCREMENT BIN
			(*mInHist.ptr<float>(RGB2HSI(bgr)))++;
			//INCREMENT NUMBER OF USED POINTS
			mUsedPts[0]++;
		}

		for(k = 0; k < mnPts; k++)
		{
			xp = *mPx.ptr<float>(0, (mnParticles + i)*mnPts + k);
			yp = *mPx.ptr<float>(1, (mnParticles + i)*mnPts + k);

			if(xp > (uint)im.cols || yp > (uint)im.rows)
				continue;
			bgr = im.at<cv::Vec3b>(cv::Point(xp, yp));
			//INCREMENT BIN
			(*mOutHist.ptr<float>(RGB2HSI(bgr)))++;
			//INCREMENT NUMBER OF USED POINTS
			mUsedPts[1]++;
		}
		
		//NORMALIZE HISTOGRAMS, IF NO USED POINTS SET WEIGHTS TO ZERO
		if(mUsedPts[0] && mUsedPts[1]) 
		{
			mInHist /= mUsedPts[0], mOutHist /= mUsedPts[1];

			//COMPUTE SQUARE ROOT OF HISTOGRAMS	
			cv::sqrt(mHists, mHists);

			float lk = 
				exp(-mG*(1.0f - (cv::sum(mInHist.mul(mModHist))[0] - 
				mD*cv::sum(mInHist.mul(mOutHist))[0] + mD)/(1 + mD)));
		
			//UPDATE WEIGHT	& ADD TO SUM OF WEIGHTS
			*mW.ptr<float>(0, i) *= lk, sum_lk += *mW.ptr<float>(0, i);
		}
		else
			*mW.ptr<float>(0, i) = 0.0f;
	}
	return sum_lk;
}
