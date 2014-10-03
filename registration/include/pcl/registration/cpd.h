#ifndef PCL_CPD_H_
#define PCL_CPD_H_

#include <pcl/pcl_base.h>


namespace pcl
{


	template <typename PointSource, typename PointTarget>
	class CoherentPointDriftAffine : public PCLBase<PointSource>
	{

		public:

			typedef pcl::PointCloud<PointSource> PointCloudSource;
			typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

			typedef pcl::PointCloud<PointTarget> PointCloudTarget;
			typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

			CoherentPointDriftAffine ()
				: max_iterations_ (150)
				, tolerance_stopping_criterion_ (1e-5)
				, outlier_weight_ (0.1)
				, use_fast_gauss_transform_ (false)
				, estimate_correspondence_ (false)
			{}

			virtual ~CoherentPointDriftAffine () {};

			PointCloudSourceConstPtr getInputSource () const;
			void setInputSource (const PointCloudSourceConstPtr &cloud);
			PointCloudTargetConstPtr getInputTarget () const;
			void setInputTarget (const PointCloudTargetConstPtr &cloud);

			void
			align(PointCloudSource &output);

		private:
			void
			transformWithGaussian(const Eigen::MatrixXf &X,
														const Eigen::MatrixXf &T,
														float sigma2,
														Eigen::VectorXf &Pt1,
														Eigen::VectorXf &P1,
														Eigen::MatrixXf &PX,
														float &L);

			int max_iterations_;
			float tolerance_stopping_criterion_;
			float outlier_weight_;
			bool use_fast_gauss_transform_;
			bool estimate_correspondence_;

	};


}


#include <pcl/registration/impl/cpd.hpp>


#endif //#ifndef PCL_CPD_H_
