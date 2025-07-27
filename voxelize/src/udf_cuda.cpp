#include <torch/extension.h>

void compute_valid_udf_cuda(float* vertices, int* faces, int* udf, const int numTriangles, const int DIM=512, const float threshold=8);

extern "C" 
void compute_valid_udf_wrapper(torch::Tensor vertices, torch::Tensor faces, torch::Tensor udf, const int numTriangles, const int DIM=512, const float threshold=8.0) {
    compute_valid_udf_cuda(vertices.data_ptr<float>(), faces.data_ptr<int>(), udf.data_ptr<int>(), numTriangles, DIM, threshold);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_valid_udf", &compute_valid_udf_wrapper, "Compute UDF using CUDA");
}

