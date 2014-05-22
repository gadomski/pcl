/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_CPD_H_
#define PCL_CPD_H_

// PCL includes
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/default_convergence_criteria.h>

namespace pcl
{
  /** \brief @b IterativeClosestPoint provides a base implementation of the Iterative Closest Point algorithm. 
    * The transformation is estimated based on Singular Value Decomposition (SVD).
    *
    * The algorithm has several termination criteria:
    *
    * <ol>
    * <li>Number of iterations has reached the maximum user imposed number of iterations (via \ref setMaximumIterations)</li>
    * <li>The epsilon (difference) between the previous transformation and the current estimated transformation is smaller than an user imposed value (via \ref setTransformationEpsilon)</li>
    * <li>The sum of Euclidean squared errors is smaller than a user defined threshold (via \ref setEuclideanFitnessEpsilon)</li>
    * </ol>
    *
    *
    * Usage example:
    * \code
    * IterativeClosestPoint<PointXYZ, PointXYZ> icp;
    * // Set the input source and target
    * icp.setInputCloud (cloud_source);
    * icp.setInputTarget (cloud_target);
    *
    * // Set the max correspondence distance to 5cm (e.g., correspondences with higher distances will be ignored)
    * icp.setMaxCorrespondenceDistance (0.05);
    * // Set the maximum number of iterations (criterion 1)
    * icp.setMaximumIterations (50);
    * // Set the transformation epsilon (criterion 2)
    * icp.setTransformationEpsilon (1e-8);
    * // Set the euclidean distance difference epsilon (criterion 3)
    * icp.setEuclideanFitnessEpsilon (1);
    *
    * // Perform the alignment
    * icp.align (cloud_source_registered);
    *
    * // Obtain the transformation that aligned cloud_source to cloud_source_registered
    * Eigen::Matrix4f transformation = icp.getFinalTransformation ();
    * \endcode
    *
    * \author Radu B. Rusu, Michael Dixon
    * \ingroup registration
    */
  template <typename PointSource, typename PointTarget, typename Scalar = float>
  class CoherentPointDrift : public Registration<PointSource, PointTarget, Scalar>
  {
    public:

      typedef enum RegistrationMode
      {
        RM_RIGID    = 0,
        RM_AFFINE   = 1, 
        RM_NONRIGID = 2, 
        LT_LENGTH   = 3
      } RegistrationMode;

      typedef typename Registration<PointSource, PointTarget, Scalar>::PointCloudSource PointCloudSource;
      typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
      typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

      typedef typename Registration<PointSource, PointTarget, Scalar>::PointCloudTarget PointCloudTarget;
      typedef typename PointCloudTarget::Ptr PointCloudTargetPtr;
      typedef typename PointCloudTarget::ConstPtr PointCloudTargetConstPtr;

      typedef PointIndices::Ptr PointIndicesPtr;
      typedef PointIndices::ConstPtr PointIndicesConstPtr;

      typedef boost::shared_ptr<CoherentPointDrift<PointSource, PointTarget, Scalar> > Ptr;
      typedef boost::shared_ptr<const CoherentPointDrift<PointSource, PointTarget, Scalar> > ConstPtr;

      using Registration<PointSource, PointTarget, Scalar>::reg_name_;
      using Registration<PointSource, PointTarget, Scalar>::getClassName;
      using Registration<PointSource, PointTarget, Scalar>::setInputSource;
      using Registration<PointSource, PointTarget, Scalar>::input_;
      using Registration<PointSource, PointTarget, Scalar>::indices_;
      using Registration<PointSource, PointTarget, Scalar>::target_;
      using Registration<PointSource, PointTarget, Scalar>::nr_iterations_;
      using Registration<PointSource, PointTarget, Scalar>::max_iterations_;
      using Registration<PointSource, PointTarget, Scalar>::previous_transformation_;
      using Registration<PointSource, PointTarget, Scalar>::final_transformation_;
      using Registration<PointSource, PointTarget, Scalar>::transformation_;
      using Registration<PointSource, PointTarget, Scalar>::transformation_epsilon_;
      using Registration<PointSource, PointTarget, Scalar>::converged_;
      using Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;
      using Registration<PointSource, PointTarget, Scalar>::inlier_threshold_;
      using Registration<PointSource, PointTarget, Scalar>::min_number_correspondences_;
      using Registration<PointSource, PointTarget, Scalar>::update_visualizer_;
      using Registration<PointSource, PointTarget, Scalar>::euclidean_fitness_epsilon_;

      typename pcl::registration::DefaultConvergenceCriteria<Scalar>::Ptr convergence_criteria_;
      typedef typename Registration<PointSource, PointTarget, Scalar>::Matrix4 Matrix4;

      /** \brief Empty constructor. */
      CoherentPointDrift () 
        : sigma2_ (-1)
        , tol_ (1.0e-5)
        , use_fgt_ (false)
        , outliers_ (0.1f)
        , use_strict_rotation_ (true)
        , scale_ (true)
        , normalize_ (true)
        , registration_mode_ (RM_RIGID)
      {
        reg_name_ = "CoherentPointDrift";
        max_iterations_ = 150;
      };

      /** \brief Empty destructor */
      virtual ~CoherentPointDrift () {}

      inline void
      setSigmaSquared (float sigma2) { sigma2_ = sigma2; }

      inline float
      getSigmaSquared () const { return (sigma2_); }

      inline void
      setTolerance (float tol) { tol_ = tol; }

      inline float
      getTolerance () const { return (tol_); }

      inline void
      setUseFgt (bool use_fgt) { use_fgt_ = use_fgt; }

      inline bool
      getUseFgt () const { return (use_fgt_); }

      inline void
      setOutlierWeight (float outliers) { outliers_ = outliers; }

      inline float
      getOutlierWeight () const { return (outliers_); }

      inline void
      setUseStrictRotation (bool use_strict_rotation) { use_strict_rotation_ = use_strict_rotation; }

      inline bool
      getUseStrictRotation () const { return (use_strict_rotation_); }

      inline void
      setEstimateScaling (bool scale) { scale_ = scale; }

      inline bool
      getEstimateScaling () const { return (scale_); }

      inline void
      setNormalize (bool normalize) { normalize_ = normalize; }

      inline bool
      getNormalize () const { return (normalize_); }

      inline void
      setRegistrationMode (RegistrationMode registration_mode) { registration_mode_ = registration_mode; }

      inline RegistrationMode
      getRegistrationMode () const { return (registration_mode_); }

    protected:

      /** \brief Rigid transformation computation method  with initial guess.
        * \param output the transformed input point cloud dataset using the rigid transformation found
        * \param guess the initial guess of the transformation to compute
        */
      virtual void 
      computeTransformation (PointCloudSource &output, const Matrix4 &guess);

      float sigma2_;
      float tol_;
      bool use_fgt_;
      float outliers_;
      bool use_strict_rotation_;
      bool scale_;
      bool normalize_;
      RegistrationMode registration_mode_;
  };
}

#include <pcl/registration/impl/cpd.hpp>

#endif  //#ifndef PCL_CPD_H_
