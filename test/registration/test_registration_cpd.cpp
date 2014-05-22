/*
* Software License Agreement (BSD License)
*
*  Point Cloud Library (PCL) - www.pointclouds.org
*  Copyright (c) 2010-2011, Willow Garage, Inc.
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

#include <gtest/gtest.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/cpd.h>

using namespace pcl;
using namespace pcl::io;
using namespace std;

PointCloud<PointXYZ> cloud_source_rigid, cloud_target_rigid, cloud_reg;
PointCloud<PointXYZ> cloud_source_affine, cloud_target_affine;
PointCloud<PointXYZRGBA> cloud_with_color;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (CPD_RIGID, CoherentPointDrift)
{
  CoherentPointDrift<PointXYZ, PointXYZ> reg;
  PointCloud<PointXYZ>::ConstPtr source (cloud_source_rigid.makeShared ());

  reg.setInputSource (source);
  reg.setInputTarget (cloud_target_rigid.makeShared ());
  reg.setRegistrationMode(reg.RM_RIGID);

  // Register
  reg.align (cloud_reg);

  FILE* fp = fopen ("C:/Projects/lidar/Code/third_party/CoherentPointDrift/output-rigid.dat", "wb");
  
  for (int i = 0; i < cloud_reg.size (); i++)
  {
    fwrite (&cloud_reg[i].x, sizeof(float), 1, fp);
    fwrite (&cloud_reg[i].y, sizeof(float), 1, fp);
    fwrite (&cloud_reg[i].z, sizeof(float), 1, fp);
  }

  fclose (fp);

  EXPECT_EQ (int (cloud_reg.points.size ()), int (cloud_source_rigid.points.size ()));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
TEST (CPD_AFFINE, CoherentPointDrift)
{
  CoherentPointDrift<PointXYZ, PointXYZ> reg;
  PointCloud<PointXYZ>::ConstPtr source (cloud_source_affine.makeShared ());

  reg.setInputSource (source);
  reg.setInputTarget (cloud_target_affine.makeShared ());
  reg.setRegistrationMode(reg.RM_AFFINE);

  // Register
  reg.align (cloud_reg);

  FILE* fp = fopen ("C:/Projects/lidar/Code/third_party/CoherentPointDrift/output-affine.dat", "wb");
  
  for (int i = 0; i < cloud_reg.size (); i++)
  {
    fwrite (&cloud_reg[i].x, sizeof(float), 1, fp);
    fwrite (&cloud_reg[i].y, sizeof(float), 1, fp);
    fwrite (&cloud_reg[i].z, sizeof(float), 1, fp);
  }

  fclose (fp);

  EXPECT_EQ (int (cloud_reg.points.size ()), int (cloud_source_affine.points.size ()));
}

/* ---[ */
int
main (int argc, char** argv)
{

  FILE* fp = fopen ("C:/Projects/lidar/Code/third_party/CoherentPointDrift/input-rigid.dat", "rb");

  cloud_source_rigid.height = 1;
  cloud_source_rigid.width = 91;
  cloud_source_rigid.is_dense = true;
  cloud_source_rigid.resize (91);

  for (int i = 0; i < cloud_source_rigid.size (); i++)
  {
    fread (&cloud_source_rigid[i].x, sizeof(float), 1, fp);
    fread (&cloud_source_rigid[i].y, sizeof(float), 1, fp);
    cloud_source_rigid[i].z = 0;
  }

  fclose (fp);


  fp = fopen ("C:/Projects/lidar/Code/third_party/CoherentPointDrift/target-rigid.dat", "rb");

  cloud_target_rigid.height = 1;
  cloud_target_rigid.width = 91;
  cloud_target_rigid.is_dense = true;
  cloud_target_rigid.resize (91);

  for (int i = 0; i < cloud_target_rigid.size (); i++)
  {
    fread (&cloud_target_rigid[i].x, sizeof(float), 1, fp);
    fread (&cloud_target_rigid[i].y, sizeof(float), 1, fp);
    cloud_target_rigid[i].z = 0;
  }

  fclose (fp);

  fp = fopen ("C:/Projects/lidar/Code/third_party/CoherentPointDrift/input-affine.dat", "rb");

  cloud_source_affine.height = 1;
  cloud_source_affine.width = 91;
  cloud_source_affine.is_dense = true;
  cloud_source_affine.resize (91);

  for (int i = 0; i < cloud_source_affine.size (); i++)
  {
    fread (&cloud_source_affine[i].x, sizeof(float), 1, fp);
    fread (&cloud_source_affine[i].y, sizeof(float), 1, fp);
    cloud_source_affine[i].z = 0;
  }

  fclose (fp);


  fp = fopen ("C:/Projects/lidar/Code/third_party/CoherentPointDrift/target-affine.dat", "rb");

  cloud_target_affine.height = 1;
  cloud_target_affine.width = 91;
  cloud_target_affine.is_dense = true;
  cloud_target_affine.resize (91);

  for (int i = 0; i < cloud_target_affine.size (); i++)
  {
    fread (&cloud_target_affine[i].x, sizeof(float), 1, fp);
    fread (&cloud_target_rigid[i].y, sizeof(float), 1, fp);
    cloud_target_affine[i].z = 0;
  }

  fclose (fp);

  testing::InitGoogleTest (&argc, argv);
  return (RUN_ALL_TESTS ());
}
/* ]--- */
