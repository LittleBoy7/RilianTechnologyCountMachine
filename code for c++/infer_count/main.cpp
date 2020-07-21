#pragma warning(disable : 4996)
#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

using namespace cv;
using namespace std;

cv::Mat ClearBlackEdge(Mat &mat_in) {
	cout << "start clearing black edge" << endl;
	int x = mat_in.cols;
	int y = mat_in.rows;

	std::vector< int>edges_x;
	std::vector<int>edges_y;
	std::vector<int>edges_x_up;
	std::vector<int>edges_y_up;
	std::vector<int>edges_x_down;
	std::vector<int>edges_y_down;
	std::vector<int>edges_x_left;
	std::vector<int>edges_y_left;
	std::vector<int>edges_x_right;
	std::vector<int>edges_y_right;

	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			if (mat_in.at<uchar>(j, i) > 250) {
				edges_x_left.push_back(i);
				edges_y_left.push_back(j);
			}
		}
		if (edges_x_left.size() != 0 || edges_y_left.size() != 0) {
			break;
		}
	}
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			if (mat_in.at<uchar>(j, x - i - 1) > 250) {
				edges_x_right.push_back(i);
				edges_y_right.push_back(j);
			}
		}
		if (edges_x_right.size() != 0 || edges_y_right.size() != 0) {
			break;
		}
	}
	for (int j = 0; j < y; j++) {
		for (int i = 0; i < x; i++) {
			if (mat_in.at<uchar>(j, i) > 250) {
				edges_x_up.push_back(i);
				edges_y_up.push_back(j);
			}
		}
		if (edges_x_up.size() != 0 || edges_y_up.size() != 0) {
			break;
		}
	}
	for (int j = 0; j < y; j++) {
		for (int i = 0; i < x; i++) {
			if (mat_in.at<uchar>(y - j - 1, i) > 250) {
				edges_x_down.push_back(i);
				edges_y_down.push_back(j);
			}
		}
		if (edges_x_down.size() != 0 || edges_x_down.size() != 0) {
			break;
		}
	}
	edges_x.insert(edges_x.end(), edges_x_left.begin(), edges_x_left.end());
	edges_x.insert(edges_x.end(), edges_x_right.begin(), edges_x_right.end());
	edges_x.insert(edges_x.end(), edges_x_up.begin(), edges_x_up.end());
	edges_x.insert(edges_x.end(), edges_x_down.begin(), edges_x_down.end());
	edges_y.insert(edges_y.end(), edges_y_left.begin(), edges_y_left.end());
	edges_y.insert(edges_y.end(), edges_y_right.begin(), edges_y_right.end());
	edges_y.insert(edges_y.end(), edges_y_up.begin(), edges_y_up.end());
	edges_y.insert(edges_y.end(), edges_y_down.begin(), edges_y_down.end());

	int left = *std::min_element(edges_x.begin(), edges_x.end());
	int right = *std::max_element(edges_x.begin(), edges_x.end());
	int bottom = *std::min_element(edges_y.begin(), edges_y.end());
	int top = *std::max_element(edges_y.begin(), edges_y.end());
	return mat_in(cv::Rect(left, bottom, right - left + 1, top - bottom + 1));
}

void GetCircle(Mat &mat_in, std::vector<cv::Point>&allCentor, std::vector<int>&allRadius) {
	cout << "start detecting outer circle" << endl;
	Mat new_img;
	cv::GaussianBlur(mat_in, new_img, cv::Size(9, 9), 0, 0);
	cv::threshold(new_img, new_img, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(100, 100));
	Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
	cv::morphologyEx(new_img, new_img, cv::MORPH_CLOSE, kernel);
	cv::erode(new_img, new_img, kernel2);
	std::vector<std::vector<cv::Point>> contourscontours;
	std::vector<Vec4i> hierarchy;
	cv::findContours(new_img, contourscontours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
	int width = mat_in.cols;
	int height = mat_in.rows;
	for (int i = 0; i < contourscontours.size(); i++) {
		cv::Point2f center;
		float radius;
		cv::minEnclosingCircle(contourscontours[i], center, radius);
		if (width / 3 < center.x && width / 3 * 2 > center.x && 0.1*width < radius && radius < 0.8*width) {
			allCentor.push_back(cv::Point(static_cast<int> (center.x), static_cast<int> (center.y)));
			allRadius.push_back(static_cast<int> (radius) + 100);
		}
	}
}

void ReduceLength(int &top, int &bottom, int value) {
	top += floor(value / 2);
	bottom -= ceil(value / 2);
}

void IncreaseLength(int &top, int &bottom, int value) {
	top -= floor(value / 2);
	bottom += ceil(value / 2);
}

int GetNum(int height) {
	int num = round(height / 400);
	if (num != 0) {
		num = num;
	}
	else {
		num = num + 2;
	}
	if (num % 2 == 0) {
		num = num;
	}
	else {
		num = num + 1;
	}
	return num;
}

cv::Mat GetSquare(Mat &mat_in, std::vector<cv::Point>&allCentor, std::vector<int>&allRadius, int &number, Point &center, int &radius) {
	cout << "start detecting outer square" << endl;
	center = allCentor[0];
	radius = allRadius[0];
	int left = 0;
	int right = mat_in.cols;
	int top = 0;
	int bottom = mat_in.rows;

	if (center.x - radius > 0) {
		left = center.x - radius;
	}
	if (center.x + radius < mat_in.cols) {
		right = center.x + radius;
	}
	if (center.y - radius > 0) {
		top = center.y - radius;
	}
	if (center.y + radius < mat_in.rows) {
		bottom = center.y + radius;
	}

	int diff = 0;
	if ((right - left) < (bottom - top)) {
		diff = ((bottom - top) - (right - left));
		ReduceLength(top, bottom, diff);
	}
	else if ((right - left) > (bottom - top)) {
		diff = ((right - left) - (bottom - top));
		ReduceLength(left, right, diff);
	}

	number = GetNum(right - left);
	diff = static_cast<int>(ceil((right - left) / (8.0 * number))) * (8 * number) - (right - left);
	IncreaseLength(left, right, diff);
	diff = static_cast<int>(ceil((bottom - top) / (8.0 * number))) * (8 * number) - (bottom - top);
	IncreaseLength(top, bottom, diff);

	int flag = 0;
	if (right > mat_in.cols || bottom > mat_in.rows) {
		flag = 8 * number * static_cast<int>(ceil(max(right - mat_in.cols, bottom - mat_in.rows) / (8.0 * number)));
	}

	int length = min(right - left - flag, bottom - top - flag);
	if ((right - left - flag) != (bottom - top - flag)) {
		if (length % 2 != 0) {
			length += 1;
		}
	}
	mat_in = mat_in(cv::Rect(left, top, length, length));
	center = Point(center.x - left, center.y - top);
	return mat_in;
}

int Crop(Mat &mat_in, int padding, int number, std::vector<cv::Mat>&all_crops) {
	cout << "start croping pinctures" << endl;
	int sum_rows = mat_in.rows;
	int sum_cols = mat_in.cols;
	int cols = static_cast<int>(sum_cols / number);
	int rows = static_cast<int>(sum_rows / number);

	for (int i = 0; i < number; i++) {
		for (int j = 0; j < number; j++) {
			int top = j * rows;
			int bottom = (j + 1) *rows;
			int left = i * cols;
			int right = (i + 1)*cols;
			if (j > 0) {
				top -= padding;
			}
			else {
				top -= 0;
			}
			if (j < number - 1) {
				bottom += padding;
			}
			else {
				bottom += 0;
			}
			if (i > 0) {
				left -= padding;
			}
			else {
				left -= 0;
			}
			if (i < number - 1) {
				right += padding;
			}
			else {
				right += 0;
			}
			Mat out_img = mat_in(cv::Rect(left, top, right - left, bottom - top));
			all_crops.push_back(out_img);
		}
	}
	return rows;
}

class Infer{
public:
	Infer();
	~Infer();

	at::Tensor inferImage(std::vector<torch::jit::IValue>& inputs);

private:
	std::shared_ptr<torch::jit::script::Module> module;
};

Infer::Infer() {
	module = torch::jit::load("./torch_script_eval.pt");

	torch::DeviceType device_type;
	device_type = torch::kCUDA;
	torch::Device device(device_type, 0);

	module->to(device);
}

Infer::~Infer() {}

at::Tensor Infer::inferImage(std::vector<torch::jit::IValue>& inputs){
	return module->forward(inputs).toTensor();
}

std::vector<cv::Mat> InferProcess(std::vector<cv::Mat> &all_crops) {
	cout << "start infering pictures" << endl;
	std::vector<cv::Mat> results;

	for (int i = 0; i < all_crops.size(); i++) {
		auto& binImg = all_crops[i];

		int H = binImg.rows;
		int W = binImg.cols;

		binImg.convertTo(binImg, CV_32FC3, 1.0f / 255.0f);
		binImg = (binImg - 0.5) / 0.5;

		Mat frame = Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
		std::vector<cv::Mat> channels;
		for (int i = 0; i < 3; i++){
			channels.push_back(binImg);
		}
		merge(channels, frame);
		
		auto input_tensor = torch::from_blob(frame.data, { 1, H, W, 3 });
		input_tensor = input_tensor.permute({ 0, 3, 1, 2 });

		torch::DeviceType device_type;
		device_type = torch::kCUDA;
		torch::Device device(device_type, 0);

		std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("./torch_script_eval.pt");
		module->to(device);

		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(input_tensor.to(device));

		at::Tensor out_tensor = module->forward(inputs).toTensor();
		out_tensor = at::_th_view(out_tensor, { out_tensor.size(2), out_tensor.size(3) });
		out_tensor = out_tensor.to(torch::kCPU);
		cv::Mat resultImg(H, W, CV_32FC1);
		std::memcpy((void *)resultImg.data, out_tensor.data_ptr(), H * W * sizeof(float));

		results.emplace_back(resultImg);

		inputs.clear();
	}
	return results;
}

cv::Mat Merge(vector<cv::Mat>& out, int num, int padding) {
	cout << "start merging pictures" << endl;
	int row = num;
	int col = num;
	int transition = 16;

	std::vector<cv::Mat>row_out;
	int length = out.front().rows - padding;
	int top_row = static_cast<int> (out.front().rows - padding - transition);
	int left_col = static_cast<int> (out.front().cols - padding - transition);
	Mat all_out;

	for (int i = 0; i < row; i++) {
		row_out.push_back(out[i * col](cv::Rect(0, 0, out[i * col].cols, top_row)));
		for (int j = 0; j < col - 1; j++) {
			int mid1_up, mid1_down;
			if (j == 0) {
				mid1_up = static_cast<int> (length - transition);
				mid1_down = static_cast<int> (length + transition);
			}
			else {
				mid1_up = static_cast<int> (length + padding - transition);
				mid1_down = static_cast<int> (length + padding + transition);
			}

			int mid2_up = static_cast<int> (padding - transition);
			int mid2_down = static_cast<int> (padding + transition);

			Mat row_mid1 = out[i * col + j](cv::Rect(0, mid1_up, out[i * col + j].cols, mid1_down - mid1_up));
			Mat row_mid2 = out[i * col + j + 1](cv::Rect(0, mid2_up, out[i * col + j + 1].cols, mid2_down - mid2_up));
			Mat row_mid = row_mid1 + row_mid2;

			int down_up = static_cast<int> (padding + transition);
			int down_down;
			if (j < col - 2) {
				down_down = static_cast<int> (length + padding - transition);
			}
			else {
				down_down = static_cast<int>(length + padding);
			}

			Mat row_down = out[i * col + j + 1](cv::Rect(0, down_up, out[i * col + j + 1].cols, down_down - down_up));
			vconcat(row_out[i], row_mid, row_out[i]);
			vconcat(row_out[i], row_down, row_out[i]);
		}

		if (i == 0) {
			all_out = (row_out[0](cv::Rect(0, 0, left_col, row_out[0].rows)));
		}
		else if (i > 0) {
			int mid1_left, mid1_right;
			if (i == 1) {
				mid1_left = static_cast<int> (length - transition);
				mid1_right = static_cast<int> (length + transition);
			}
			else {
				mid1_left = static_cast<int> (length + padding - transition);
				mid1_right = static_cast<int> (length + padding + transition);
			}

			int mid2_left = static_cast<int> (padding - transition);
			int mid2_right = static_cast<int> (padding + transition);
			Mat col_mid1 = row_out[i - 1](cv::Rect(mid1_left, 0, mid1_right - mid1_left, row_out[i - 1].rows));
			Mat col_mid2 = row_out[i](cv::Rect(mid2_left, 0, mid2_right - mid2_left, row_out[i].rows));
			Mat col_mid = 0.2 * col_mid1 + 0.8 * col_mid2;

			int right_left = static_cast<int> (padding + transition);
			int right_right;
			if (i < row - 1) {
				right_right = static_cast<int> (length + padding - transition);
			}
			else {
				right_right = static_cast<int> (length + padding);
			}
			Mat col_right = row_out[i](cv::Rect(right_left, 0, right_right - right_left, row_out[i].rows));
			hconcat(all_out, col_mid, all_out);
			hconcat(all_out, col_right, all_out);
		}
	}
	return all_out;
}

cv::Mat DrawCircle(Mat& result, Mat& all_img, Point &center, int &radius) {
	cout << "start drawing circles" << endl;
	cout << endl;

	Mat points = result * 100;
	Mat out;
	cv::threshold(points, out, 3.5, 255, cv::THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
	if (sum(result)[0] < 300) {
		dilate(out, out, element, Point(-1, -1), 3);
	}
	else {
		dilate(out, out, element, Point(-1, -1), 1);
	}
	out.convertTo(out, CV_8UC1);
	cvtColor(all_img, all_img, CV_GRAY2BGR);

	cv::Mat labels, stats, centorids;
	Point location;
	string title = "Number: ";

	int nLabels = cv::connectedComponentsWithStats(out, labels, stats, centorids, 4, 4);
	float flag = 0;
	int count = 0;
	for (int i = 1; i < nLabels; i++) {
		location = Point(static_cast<int> (centorids.at<double>(i, 0)), static_cast<int> (centorids.at<double>(i, 1)));
		flag = sqrt((location.x - center.x) * (location.x - center.x) + (location.y - center.y) * (location.y - center.y));
		if (flag < radius && flag > 300) {
			cv::circle(all_img, location, 3, (255, 255, 255), -1);
			count += 1;
		}
	}
	cv::putText(all_img, title.append(to_string(count)), Point(center.x - 300, center.y), CV_FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 3);
	return all_img;
}

cv::Mat Getresult(Mat &mat_in) {
	int padding = 80;
	int number = 0;
	int radius = 0;
	Point center = Point(0, 0);

	Mat out_img = ClearBlackEdge(mat_in);
	std::vector<cv::Point>allCentor;
	std::vector<int>allRadius;
	GetCircle(out_img, allCentor, allRadius);
	Mat all_img = GetSquare(out_img, allCentor, allRadius, number, center, radius);

	std::vector<cv::Mat> all_crops;
	int rows = Crop(all_img, padding, number, all_crops);

	std::vector<cv::Mat> results = InferProcess(all_crops);

	Mat out = Merge(results, number, padding);
	out = DrawCircle(out, all_img, center, radius);
	
	return out;
}

void ReadImages(cv::String pattern, vector<Mat> &images, vector<string> &filename)
{
	vector<cv::String> fn;
	glob(pattern, fn, false);

	size_t count = fn.size();
	for (size_t i = 0; i < count; i++)
	{
		images.push_back(imread(fn[i], 0));
		filename.push_back(fn[i].substr(17));
	}
}
cv::Mat InferceAllMaps(Mat &mat_in) {
	return  Getresult(mat_in);   // mat_in必须是单通道的大图,调用的话调用这个接口
}
int main()   //测试使用
{
	cv::String pattern = "F:/sh/39test_img/*.jpg";
	string savedir = "F:/sh/test/";
	string savepath;
	vector<Mat> images;
	vector<string> filename;
	ReadImages(pattern, images, filename);

	for (int i = 0; i < images.size(); i++) {
		savepath = savedir;
		savepath = savepath.append(filename[i]);
		cout << "processing " << savepath << endl;
		auto result = Getresult(images[i]);
		imwrite(savepath, result);
	}
}
