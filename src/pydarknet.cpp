#include <iostream>
#include <boost/python.hpp>
#include <Python.h>
#include <vector>

#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"

#ifdef GPU
#include "cuda.h"
#endif

using namespace std;
namespace bp = boost::python;

string voc_class_names[] = {
	"aeroplane", "bicycle", "bird", "boat", "bottle",
	"bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant",
	"sheep", "sofa", "train", "tvmonitor" };

typedef struct BBox{
	int left;
	int right;
	int top;
	int bottom;
	float confidence;
	int cls;
} _BBox;

void draw_detections_bbox(image im, int num, float thresh, box *boxes, float **probs, string* names, image *labels, int classes, vector<_BBox> &bb)
{
	int i;

	for (i = 0; i < num; ++i){
		int classs = max_index(probs[i], classes);
		float prob = probs[i][classs];
		if (prob > thresh){
			int width = pow(prob, 1. / 2.) * 10 + 1;
			int offset = classs * 17 % classes;
			float red = get_color(0, offset, classes);
			float green = get_color(1, offset, classes);
			float blue = get_color(2, offset, classes);
			float rgb[3];
			rgb[0] = red;
			rgb[1] = green;
			rgb[2] = blue;
			box b = boxes[i];

			int left = (b.x - b.w / 2.)*im.w;
			int right = (b.x + b.w / 2.)*im.w;
			int top = (b.y - b.h / 2.)*im.h;
			int bot = (b.y + b.h / 2.)*im.h;

			if (left < 0) left = 0;
			if (right > im.w - 1) right = im.w - 1;
			if (top < 0) top = 0;
			if (bot > im.h - 1) bot = im.h - 1;

			//printf("%s: %.2f, %d, %d, %d, %d\n", names[classs].c_str(), prob, left, right, top, bot);
			_BBox bs;
			bs.left = left; bs.right = right; bs.top = top; bs.bottom = bot; bs.confidence = prob; bs.cls = classs;
			bb.push_back(bs);

			draw_box_width(im, left, top, right, bot, width, red, green, blue);
			if (labels) draw_label(im, top + width, left, labels[classs], rgb);
		}
	}
}

void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
	int i, j, n;
	//int per_cell = 5*num+classes;
	for (i = 0; i < side*side; ++i){
		int row = i / side;
		int col = i % side;
		for (n = 0; n < num; ++n){
			int index = i*num + n;
			int p_index = side*side*classes + i*num + n;
			float scale = predictions[p_index];
			int box_index = side*side*(classes + num) + (i*num + n) * 4;
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
			boxes[index].y = (predictions[box_index + 1] + row) / side * h;
			boxes[index].w = pow(predictions[box_index + 2], (square ? 2 : 1)) * w;
			boxes[index].h = pow(predictions[box_index + 3], (square ? 2 : 1)) * h;

			//printf("scale : %f, %f, %f, %f, %f\n", scale, boxes[index].x, boxes[index].y, boxes[index].w, boxes[index].h);

			for (j = 0; j < classes; ++j){
				int class_index = i*classes;
				float prob = scale*predictions[class_index + j];
				probs[index][j] = (prob > thresh) ? prob : 0;
			}
			if (only_objectness){
				probs[index][0] = scale;
			}
		}
	}
}

class DarknetObjectDetector{
public:
	DarknetObjectDetector(bp::str cfg_name, bp::str weight_name){

		string cfg_c_name = string(((const char*)bp::extract<const char*>(cfg_name)));
		string weight_c_name = string(((const char*)bp::extract<const char*>(weight_name)));
		cout << "loading network spec from" << cfg_c_name << '\n';
		net = parse_network_cfg((char*)cfg_c_name.c_str());

		cout << "loading network weights from" << weight_c_name << '\n';
		load_weights(&net, (char*)weight_c_name.c_str());

		cout << "network initialized!\n";
		layer = get_network_detection_layer(net);
		set_batch_network(&net, 1); srand(2222222);

		thresh = 0.2;
		boxes = (box *)calloc(layer.side*layer.side*layer.n, sizeof(box));
		probs = (float **)calloc(layer.side*layer.side*layer.n, sizeof(float *));
		for (int j = 0; j < layer.side*layer.side*layer.n; j++)
		{
			probs[j] = (float *)calloc(layer.classes, sizeof(float));
		}
	};

	~DarknetObjectDetector()
	{
		free(boxes);
		for (int j = 0; j < layer.side*layer.side*layer.n; j++)
		{
			free(probs[j]);
		}
		free(probs);
	};

	bp::list detect_object(bp::str img_data, int img_width, int img_height, int img_channel){

		// preprocess input image
		const unsigned char* data = (const unsigned char*)((const char*)bp::extract<const char*>(img_data));
		bp::list ret_list = bp::list();
		vector<_BBox> bboxes;

		assert(img_channel == 3);
		image im = make_image(img_width, img_height, img_channel);

		int cnt = img_height * img_channel * img_width;
		for (int i = 0; i < cnt; ++i){
			im.data[i] = (float)data[i] / 255.;
		}

		image sized = resize_image(im, net.w, net.h);
		float *X = sized.data;
		float *predictions = network_predict(net, X);
		float nms = .5f;

		convert_yolo_detections(predictions, layer.classes, layer.n, layer.sqrt, layer.side, 1, 1, thresh, probs, boxes, 1);
		if (nms) do_nms_sort(boxes, probs, layer.side*layer.side*layer.n, layer.classes, nms);
		draw_detections_bbox(im, layer.side*layer.side*layer.n, thresh, boxes, probs, voc_class_names, 0, 20, bboxes);

		save_image(im, "temp");

		free_image(im);
		free_image(sized);

		for (int i = 0; i < bboxes.size(); i++)
		{
			ret_list.append<BBox>(bboxes[i]);
		}


		return ret_list;
		//return parse_yolo_detection(predictions, 7, layer.objectness, thresh, im.w, im.h);
	};

	static void set_device(int dev_id){
#ifdef GPU
		cudaError_t err = cudaSetDevice(dev_id);
		if (err != cudaSuccess){
			cout << "CUDA Error on setting device: " << cudaGetErrorString(err) << '\n';
			PyErr_SetString(PyExc_Exception, "Not able to set device");
		}
#else
		PyErr_SetString(PyExc_Exception, "Not compiled with CUDA");
#endif
	}

private:

	bp::list parse_yolo_detection(float *box, int side,
		int objectness, float thresh,
		int im_width, int im_height)
	{
		int classes = 20;
		int elems = 4 + classes + objectness;
		int j;
		int r, c;

		bp::list ret_list = bp::list();

		for (r = 0; r < side; ++r){
			for (c = 0; c < side; ++c){
				j = (r*side + c) * elems;
				float scale = 1;
				if (objectness) scale = 1 - box[j++];
				int cls = max_index(box + j, classes);
				if (scale * box[j + cls] > thresh){
					//valid detection over threshold
					float conf = scale * box[j + cls];
					//            printf("%f %s\n", conf, voc_class_names[cls].c_str());

					j += classes;
					float x = box[j + 0];
					float y = box[j + 1];
					x = (x + c) / side;
					y = (y + r) / side;
					float w = box[j + 2]; //*maxwidth;
					float h = box[j + 3]; //*maxheight;
					h = h*h;
					w = w*w;

					int left = (x - w / 2)*im_width;
					int right = (x + w / 2)*im_width;
					int top = (y - h / 2)*im_height;
					int bottom = (y + h / 2)*im_height;

					BBox bbox = { left, right, top, bottom, conf, cls };
					ret_list.append<BBox>(bbox);
				}
			}
		}

		return ret_list;
	}

	network net;
	detection_layer layer;
	float thresh;
	box *boxes;
	float **probs;
};

BOOST_PYTHON_MODULE(libpydarknet)
{
	bp::class_<DarknetObjectDetector>("DarknetObjectDetector", bp::init<bp::str, bp::str>())
		.def("detect_object", &DarknetObjectDetector::detect_object)
		.def("set_device", &DarknetObjectDetector::set_device)
		.staticmethod("set_device");

	bp::class_<BBox>("BBox")
		.def_readonly("left", &BBox::left)
		.def_readonly("right", &BBox::right)
		.def_readonly("top", &BBox::top)
		.def_readonly("bottom", &BBox::bottom)
		.def_readonly("confidence", &BBox::confidence)
		.def_readonly("cls", &BBox::cls);
}
