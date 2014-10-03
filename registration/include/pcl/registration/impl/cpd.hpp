#ifndef PCL_REGISTRATION_IMPL_CPD_HPP_
#define PCL_REGISTRATION_IMPL_CPD_HPP_


template <typename PointSource, typename PointTarget>
void
pcl::CoherentPointDriftAffine<PointSource, PointTarget>::align (
		PointCloudSource &output)
{
	size_t D = 3;
	size_t M = getInputSource()->points.size();
	size_t N = getInputTarget()->points.size();

	// Convert points to eigen matrices
	Eigen::MatrixXf X (N, D);
	Eigen::MatrixXf T (M, D);
	Eigen::MatrixXf Y (M, D);
	Eigen::VectorXf Pt1 (N);
	Eigen::VectorXf P1 (M);
	Eigen::VectorXf P (M);
	Eigen::MatrixXf PX = Eigen::MatrixXf::Zero (M, D);

	for (size_t i = 0; i < N; ++i)
	{
		X(i, 0) = getInputSource()->points[i].x;
		X(i, 1) = getInputSource()->points[i].y;
		X(i, 2) = getInputSource()->points[i].z;
	}

	for (size_t i = 0; i < M; ++i)
	{
		Y(i, 0) = getInputTarget()->points[i].x;
		Y(i, 1) = getInputTarget()->points[i].y;
		Y(i, 2) = getInputTarget()->points[i].z;
	}

	T = Y;

	float sigma2 = (M * (X.transpose() * X).trace() +
								   N * (Y.transpose() * Y).trace() -
									 2 * X.colwise().sum() * Y.colwise().sum().transpose()) /
									(M * N * D);

	int iterations = 0;
	float ntol = tolerance_stopping_criterion_ + 10.0f;
	float L = 1.0f;
	float L_old;

	while (iterations < max_iterations_ &&
				 ntol > tolerance_stopping_criterion_ &&
				 sigma2 > 10 * std::numeric_limits<float>::epsilon())
	{
		L_old = L;
		// TODO enable fast gauss transform
		transformWithGaussian(X, T, sigma2, P1, Pt1, PX, L);
		ntol = std::abs((L - L_old) / L);

		float Np = Pt1.sum();
		Eigen::VectorXf mu_x = X.transpose() * Pt1 / Np;
		Eigen::VectorXf mu_y = Y.transpose() * P1 / Np;

		Eigen::MatrixXf B1 = PX.transpose() * Y -
												 Np * (mu_x * mu_y.transpose());
		Eigen::MatrixXf B2 = Y.transpose().cwiseProduct(P1.replicate(1, D)).transpose() * Y.transpose() -
												 Np * (mu_y * mu_y.transpose());
		Eigen::MatrixXf B = B1 * B2.inverse();

		Eigen::MatrixXf t = mu_x - B * mu_y;

		sigma2 = std::abs(
				(X.array().square() * Pt1.replicate(1, D).array()).matrix().sum() -
				Np * mu_x.dot(mu_x) - 
				(B1 * B.transpose()).trace()) / (Np * D);

		T = Y * B.transpose() + t.transpose().replicate(M, 1);

		++iterations;
	}
}


#endif // #ifndef PCL_REGISTRATION_IMPL_CPD_HPP_
