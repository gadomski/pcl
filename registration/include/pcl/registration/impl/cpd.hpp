/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc
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

#ifndef PCL_REGISTRATION_IMPL_CPD_HPP_
#define PCL_REGISTRATION_IMPL_CPD_HPP_

#include <Eigen/Core>

#include <pcl/registration/boost.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename Scalar> void
pcl::CoherentPointDrift<PointSource, PointTarget, Scalar>::computeTransformation (
    PointCloudSource &output, const Matrix4 &guess)
{
  // Point cloud containing the correspondences of each point in <input, indices>
  PointCloudSourcePtr input_transformed (new PointCloudSource);

  // If the guessed transformation is non identity
  if (guess != Matrix4::Identity ())
  {
    input_transformed->resize (input_->size ());
     // Apply guessed transformation prior to search for neighbours
    pcl::transformPointCloud (*input_, *input_transformed, guess);
  }
  else
    *input_transformed = *input_;
 
  nr_iterations_ = 0;
  converged_ = false;
  final_transformation_ = guess;

  size_t D = 3; // x, y, z
  size_t N = indices_->size ();
  size_t M = target_->points.size ();
  size_t DNM = D * N * M;

  // convert input data to eigen data
  Eigen::MatrixXf X (N, D);
  Eigen::MatrixXf T (M, D);
  Eigen::MatrixXf Y (M, D);

  // copy the input data to the intput and output matrices
  for (size_t i = 0; i < N; ++i)
  {
    X (i, 0) = input_transformed->points[(*indices_)[i]].x;
    X (i, 1) = input_transformed->points[(*indices_)[i]].y;
    X (i, 2) = input_transformed->points[(*indices_)[i]].z;
  }

  // copy the target data
  for (size_t i = 0; i < M; ++i)
  {
    Y (i, 0) = target_->points[i].x;
    Y (i, 1) = target_->points[i].y;
    Y (i, 2) = target_->points[i].z;
  }

  T = Y;

  //Eigen::MatrixXf mu_x (D, 1);
  //Eigen::MatrixXf mu_y (D, 1);
  //mu_x.setZero ();
  //mu_y.setZero ();
  Eigen::MatrixXf xd = Eigen::MatrixXf::Zero(1, D);
  Eigen::MatrixXf yd = Eigen::MatrixXf::Zero(1, D);

  float x_scale = 1.0f;
  float y_scale = 1.0f;
  
  // normalize to zero mean and unit variance
  if (normalize_)
  {
    // we lose some precision here due to the use of float over double
    xd = X.colwise ().mean ();
    yd = Y.colwise ().mean ();

    X = X - xd.replicate (N, 1);
    Y = Y - yd.replicate (M, 1);

    x_scale = std::sqrt (X.array ().square ().matrix ().sum () / N);
    y_scale = std::sqrt (Y.array ().square ().matrix ().sum () / M);

    X = X / x_scale;
    Y = Y / y_scale;
  }

  // initialization stuff
  if (sigma2_ <= 0)
  {
    float trX = (X.transpose () * X).trace ();
    float trY = (Y.transpose () * Y).trace ();

    sigma2_ = (N * trY + M * trX - (2.0f * Y.colwise ().sum () * X.colwise ().sum ().transpose ())) / DNM;
  }
  
  float sigma2_init = sigma2_;

  // initialize the rotation matrix and scaling coefficient
  Eigen::MatrixXf R = Eigen::MatrixXf::Identity(D, D);
  Eigen::MatrixXf B;
  int iter = 0;
  float ntol = tol_ + 10.0f;
  float L;
  float L_old;
  
  if (registration_mode_ == RM_RIGID)
    L = 0.0f;
  else
    L = 1.0f;

  Eigen::VectorXf Pt1 (N); // Pt1
  Eigen::VectorXf P1 (M); // P1
  Eigen::VectorXf P (M);
  std::vector <float> temp_x (D); // temp_x
  Eigen::MatrixXf Px = Eigen::MatrixXf::Zero (M, D);

  float sigma2save = sigma2_;
  float s = 1;
  Eigen::VectorXf t;

  // loop until convergence
  while ((iter++ < max_iterations_) && (ntol > tol_) && (sigma2_ > 10*std::numeric_limits<float>::epsilon ()))
  {
    // clear our probability vectors
    Pt1.setZero ();
    P1.setZero ();
    P.setZero ();
    Px.setZero ();

    // save the previous likelihood estimate
    L_old = L;

    // Expectation Step
    if (use_fgt_)
    {
      float hsigma = std::sqrt (2.0f * sigma2_);
      if (outliers_ == 0)
        outliers_ = 10*std::numeric_limits<float>::epsilon ();

      float e = 9;
      Eigen::Vector3f foo;
      foo << N, M, 50*sigma2_init/sigma2_;
      float K = std::round (foo.minCoeff());
      float P = 6;

      //fgt_model
      //fgt_predict

      float ndi = outliers_ / (1-outliers_) * M / N * std::pow (2 * M_PI * sigma2_, 0.5 * D); //clean up, same as outlier in the non-FGT mode
      float denomP; // = Kt1 + ndi
      //Pt1 = 1 - ndi.cwiseQuotient (denomP);
      Pt1.transposeInPlace ();

      //fgt_model
      //fgt_predict
      
      for (size_t d = 0; d < D; ++d)
      {
        //fgt_model
        //fgt_predict
      }

      Px.transposeInPlace ();

    //  L -= denomP.array ().log ().matrix ().sum () + D * N * std::log (sigma2_) / 2;
    }
    else
    {
      float ksig = -2.0 * sigma2_;
      float outlier = (outliers_ * N * std::pow (-ksig * M_PI, 0.5 * D)) / ((1.0 - outliers_) * M);

      // reset our likelihood estimate
      L = 0.0f;

      for (int n = 0; n < N; ++n) 
      {
        float sp = 0.0;

        for (int m = 0; m < M; ++m) 
        {
          float ranz = 0.0;
          
          for (int d = 0; d < D; ++d) 
          {
            float diff = X (n, d) - Y (m, d);  
            diff = diff * diff;
            ranz += diff;
          }
          
          P (m) = std::exp (ranz / ksig);
          sp += P (m);
        }
      
        sp += outlier;
    
        Pt1 (n) = 1 - (outlier / sp);
      
        for (int d = 0; d < D; ++d) 
        {
          temp_x[d] = X (n, d) / sp;
        }
      
        for (int m = 0; m < M; ++m) 
        {
          P1 (m) += P (m) / sp;
          
          for (int d = 0; d < D; ++d) 
          {
            Px (m, d) += temp_x[d] * P[m];
          }
        }
      
        L += -std::log (sp);     
      }

      L += (float)D * (float)M * std::log (sigma2_) / 2.0f;
    }

    // keep track of the % change in likelihood
    ntol = std::abs ((L - L_old) / L);

    std::cerr << "CPD Rigid dL = " << ntol << ", iter = " << iter << ", sigma2 = " << sigma2_ << std::endl;

    // precompute
    float Np = Pt1.sum ();
    Eigen::VectorXf mu_x = X.transpose () * Pt1 / Np;
    Eigen::VectorXf mu_y = Y.transpose () * P1 / Np;

    // solve for rotation, scaling, translation and sigma_squared
    Eigen::MatrixXf A = Px.transpose () * Y - Np * (mu_x * mu_y.transpose ());

    if (registration_mode_ == RM_RIGID)
    {
      Eigen::JacobiSVD <Eigen::MatrixXf> svd (A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
      Eigen::MatrixXf C = Eigen::MatrixXf::Identity(D, D);

      if (use_strict_rotation_)
        C (D-1, D-1) = (svd.matrixU () * svd.matrixV ().transpose ()).determinant ();

      R = svd.matrixU () * C * svd.matrixV ().transpose ();

      sigma2save = sigma2_;

      if (scale_)
      {
        float t1 = (svd.singularValues ().asDiagonal () * C).trace ();
        float t2 = (Y.array ().square ().matrix ().cwiseProduct (P1.replicate (1, D))).sum();
        float t3 = Np * (float)(mu_y.transpose () * mu_y);
        s = t1 / (t2 - t3);

        t1 = (X.array ().square ().matrix ().cwiseProduct (Pt1.replicate (1, D))).sum();
        t2 = Np * (float)(mu_x.transpose () * mu_x);
        t3 = s * (svd.singularValues ().asDiagonal () * C).trace ();
        sigma2_ = std::abs(t1 - t2 - t3) / (Np * D);
      }
      else
      {
        float t1 = ((Y.array ().square ().matrix () * Pt1.replicate (D, 1)).colwise ().sum ()).sum();
        float t2 = Np * (float)(mu_x.transpose () * mu_x);
        float t3 = ((X.array ().square ().matrix () * P1.replicate (D, 1)).colwise ().sum ()).sum();
        float t4 = Np * (float)(mu_y.transpose () * mu_y);
        float t5 = 2 * (svd.singularValues ().matrix ().transpose () * C).sum ();
        sigma2_ = std::abs((t1 - t2 + t3 - t4 - t5) / (Np * D));
      }

      t = mu_x - s * R * mu_y;

      // update the GMM centroids
      T = s * Y * R.transpose () + t.transpose ().replicate (M, 1);
    }
    else if (registration_mode_ == RM_AFFINE)
    {
      //B2=(Y.*repmat(P1,1,D))'*Y-Np*(mu_y*mu_y');
      Eigen::MatrixXf t1 = (X.transpose ().cwiseProduct (P1.replicate (1, D))).transpose () * X.transpose ();
      Eigen::MatrixXf t2 = Np * (mu_y * mu_y.transpose ());
      Eigen::MatrixXf B2 = t1 - t2;
      
      //B=B1/B2; % B= B1 * inv(B2);
      B = A * B2.inverse ();

      sigma2save = sigma2_;

      //sigma2=abs(sum(sum(X.^2.*repmat(Pt1,1,D)))- Np*(mu_x'*mu_x) -trace(B1*B'))/(Np*D); 
      T = s * R * X + t.replicate (1, N);
      float f1 = ((Y.array ().square ().matrix () * Pt1.replicate (D, 1)).colwise ().sum ()).sum();
      float f2 = Np * (float)(mu_x.transpose () * mu_x);
      float f3 = (A.transpose () * B).trace ();
      sigma2_ = std::abs(f1 - f2 - f3) / (Np * D);

      t = mu_x - B * mu_y;

      // update the GMM centroids
      T = X * B.transpose () + t.replicate (1, N);
    }
  }

  // denormalize
  if (normalize_)
  {
    s = s * x_scale / y_scale;
    t = x_scale * t + xd.transpose () - s * (R * yd.transpose ());
    T = T * x_scale + xd.replicate (M, 1);

    if (registration_mode_ == RM_AFFINE)
    {
      R = s * B;
      s = 1.0f;
    }
  }

  converged_ = true; // we are not using converged_ correctly here, but it will work for now
  Eigen::AngleAxisf init_rotation;
  Eigen::Matrix3f rotation (s * R.transpose ());
  init_rotation.fromRotationMatrix(rotation);
  Eigen::Translation3f init_translation (t);
  final_transformation_ = (init_translation * init_rotation).matrix ();

  // copy output matrix to output point cloud
  for (size_t i = 0; i < M; ++i)
  {
    output.points[i].x = T (i, 0);
    output.points[i].y = T (i, 1);
    output.points[i].z = T (i, 2);
  }
}

#endif /* PCL_REGISTRATION_IMPL_CPD_HPP_ */
