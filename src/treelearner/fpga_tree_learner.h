#ifndef LIGHTGBM_TREELEARNER_FPGA_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_FPGA_TREE_LEARNER_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/array_args.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/feature_group.h>
#include "feature_histogram.hpp"
#include "serial_tree_learner.h"
#include "data_partition.hpp"
#include "split_info.hpp"
#include "leaf_splits.hpp"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <memory>

#ifdef USE_FPGA

// #define BOOST_COMPUTE_THREAD_SAFE
// #define BOOST_COMPUTE_HAVE_THREAD_LOCAL
// // Use Boost.Compute on-disk kernel cache
// #define BOOST_COMPUTE_USE_OFFLINE_CACHE
// #include <boost/compute/core.hpp>
// #include <boost/compute/container/vector.hpp>
// #include <boost/align/aligned_allocator.hpp>

#include "xcl2.hpp"
#include <CL/cl.h>

using namespace json11;

namespace LightGBM {

/*!
* \brief FPGA-based parallel learning algorithm.
*/
class FPGATreeLearner: public SerialTreeLearner {
public:
  explicit FPGATreeLearner(const Config* tree_config);
  ~FPGATreeLearner();
  void Init(const Dataset* train_data, bool is_constant_hessian) override;
  void ResetTrainingData(const Dataset* train_data) override;
  Tree* Train(const score_t* gradients, const score_t *hessians,
              bool is_constant_hessian, Json& forced_split_json) override;

  void SetBaggingData(const data_size_t* used_indices, data_size_t num_data) override {
    SerialTreeLearner::SetBaggingData(used_indices, num_data);
    // determine if we are using bagging before we construct the data partition
    // thus we can start data movement to FPGA earlier
    if (used_indices != nullptr) {
      if (num_data != num_data_) {
        use_bagging_ = true;
        return;
      }
    }
    use_bagging_ = false;
  }

protected:
  void BeforeTrain() override;
  bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf) override;
  void FindBestSplits() override;
  void Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) override;
  void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract) override;
private:
  /*! \brief 4-byte feature tuple used by FPGA kernels */
  struct Feature4 {
      uint8_t s[4];
  };
  
  /*! \brief Single precision histogram entiry for FPGA */
  struct FPGAHistogramBinEntry {
    score_t sum_gradients;
    score_t sum_hessians;
    uint32_t cnt;
  };

  /*!
  * \brief Find the best number of workgroups processing one feature for maximizing efficiency
  * \param leaf_num_data The number of data examples on the current leaf being processed
  * \return Log2 of the best number for workgroups per feature, in range 0...kMaxLogWorkgroupsPerFeature
  */
  int GetNumWorkgroupsPerFeature(data_size_t leaf_num_data);
  
  /*!
  * \brief Initialize FPGA device, context and command queues
  *        Also compiles the OpenCL kernel
  * \param platform_id OpenCL platform ID
  * \param device_id OpenCL device ID
  */
  //void InitFPGA(int platform_id, int device_id);
  void InitFPGA();

  /*!
  * \brief Allocate memory for FPGA computation
  */
  void AllocateFPGAMemory();

  /*!
  * \brief Compile OpenCL FPGA source code to kernel binaries
  */
  void BuildFPGAKernels();
  
  /*!
   * \brief Returns OpenCL kernel build log when compiled with option opts
   * \param opts OpenCL build options 
   * \return OpenCL build log
  */
  std::string GetBuildLog(const std::string &opts);

  /*!
  * \brief Setup FPGA kernel arguments, preparing for launching
  */
  void SetupKernelArguments();

  /*! 
   * \brief Compute FPGA feature histogram for the current leaf.
   *        Indices, gradients and hessians have been copied to the device.
   * \param leaf_num_data Number of data on current leaf
   * \param use_all_features Set to true to not use feature masks, with a faster kernel
  */
  void FPGAHistogram(data_size_t leaf_num_data, bool use_all_features);
  
  /*!
   * \brief Wait for FPGA kernel execution and read histogram
   * \param histograms Destination of histogram results from FPGA.
  */
  template <typename HistType>
  void WaitAndGetHistograms(HistogramBinEntry* histograms);

  /*!
   * \brief Construct FPGA histogram asynchronously. 
   *        Interface is similar to Dataset::ConstructHistograms().
   * \param is_feature_used A predicate vector for enabling each feature
   * \param data_indices Array of data example IDs to be included in histogram, will be copied to FPGA.
   *                     Set to nullptr to skip copy to FPGA.
   * \param num_data Number of data examples to be included in histogram
   * \param gradients Array of gradients for all examples.
   * \param hessians Array of hessians for all examples.
   * \param ordered_gradients Ordered gradients will be generated and copied to FPGA when gradients is not nullptr, 
   *                     Set gradients to nullptr to skip copy to FPGA.
   * \param ordered_hessians Ordered hessians will be generated and copied to FPGA when hessians is not nullptr, 
   *                     Set hessians to nullptr to skip copy to FPGA.
   * \return true if FPGA kernel is launched, false if FPGA is not used
  */
  bool ConstructFPGAHistogramsAsync(
    const std::vector<int8_t>& is_feature_used,
    const data_size_t* data_indices, data_size_t num_data,
    const score_t* gradients, const score_t* hessians,
    score_t* ordered_gradients, score_t* ordered_hessians);


  /*! brief Log2 of max number of workgroups per feature*/
  const int kMaxLogWorkgroupsPerFeature = 0; // 2^10 //default = 10
  /*! brief Max total number of workgroups with preallocated workspace.
   *        If we use more than this number of workgroups, we have to reallocate subhistograms */
  int preallocd_max_num_wg_ = 1024;

  /*! \brief True if bagging is used */
  bool use_bagging_;

  /*! \brief FPGA device object */
  //boost::compute::device dev_;
  std::vector<cl::Device> devices;
  cl::Device dev_;
  /*! \brief FPGA context object */
  //boost::compute::context ctx_;
  cl::Context ctx_;
  /*! \brief FPGA command queue object */
  //boost::compute::command_queue queue_;
  cl::CommandQueue queue_;
  // /*! \brief FPGA kernel for 256 bins */
  // const char *kernel256_src_ = 
  // #include "ocl/histogram256.cl"
  // ;
  // /*! \brief FPGA kernel for 64 bins */
  // const char *kernel64_src_ = 
  // #include "ocl/histogram64.cl"
  // ;
  // /*! \brief FPGA kernel for 16 bins */
  // const char *kernel16_src_ = 
  // #include "ocl/histogram16.cl"
  //;
  /*! \brief Currently used kernel source */
  std::string kernel_source_;
  /*! \brief Currently used kernel name */
  std::string kernel_name_;

  /*! \brief a array of histogram kernels with different number
     of workgroups per feature */
  //std::vector<boost::compute::kernel> histogram_kernels_;
  std::vector<cl::Kernel> histogram_kernels_;
  /*! \brief a array of histogram kernels with different number
     of workgroups per feature, with all features enabled to avoid branches */
  //std::vector<boost::compute::kernel> histogram_allfeats_kernels_;
  std::vector<cl::Kernel> histogram_allfeats_kernels_;
  /*! \brief a array of histogram kernels with different number
     of workgroups per feature, and processing the whole dataset */
  //std::vector<boost::compute::kernel> histogram_fulldata_kernels_;
  std::vector<cl::Kernel> histogram_fulldata_kernels_;
  /*! \brief total number of feature-groups */
  int num_feature_groups_;
  /*! \brief total number of dense feature-groups, which will be processed on FPGA */
  int num_dense_feature_groups_;
  /*! \brief On FPGA we read one DWORD (4-byte) of features of one example once.
   *  With bin size > 16, there are 4 features per DWORD.
   *  With bin size <=16, there are 8 features per DWORD.
   * */
  int dword_features_;
  /*! \brief total number of dense feature-group tuples on FPGA.
   * Each feature tuple is 4-byte (4 features if each feature takes a byte) */
  int num_dense_feature4_;
  /*! \brief Max number of bins of training data, used to determine 
   * which FPGA kernel to use */
  int max_num_bin_;
  /*! \brief Used FPGA kernel bin size (64, 256) */
  int device_bin_size_;
  /*! \brief Size of histogram bin entry, depending if single or double precision is used */
  size_t hist_bin_entry_sz_;
  /*! \brief Indices of all dense feature-groups */
  std::vector<int> dense_feature_group_map_;
  /*! \brief Indices of all sparse feature-groups */
  std::vector<int> sparse_feature_group_map_;
  /*! \brief Multipliers of all dense feature-groups, used for redistributing bins */
  std::vector<int> device_bin_mults_;
  /*! \brief FPGA memory object holding the training data */
  //std::unique_ptr<boost::compute::vector<Feature4>> device_features_;
  std::unique_ptr<cl::vector<Feature4>> device_features_;
  /*! \brief FPGA memory object holding the ordered gradient */
  //boost::compute::buffer device_gradients_;
  cl::Buffer device_gradients_;
  /*! \brief Pinned memory object for ordered gradient */
  //boost::compute::buffer pinned_gradients_;
  cl::Buffer pinned_gradients_;
  /*! \brief Pointer to pinned memory of ordered gradient */
  void * ptr_pinned_gradients_ = nullptr;
  /*! \brief FPGA memory object holding the ordered hessian */
  //boost::compute::buffer device_hessians_;
  cl::Buffer device_hessians_;
  /*! \brief Pinned memory object for ordered hessian */
  //boost::compute::buffer pinned_hessians_;
  cl::Buffer pinned_hessians_;
  /*! \brief Pointer to pinned memory of ordered hessian */
  void * ptr_pinned_hessians_ = nullptr;
  /*! \brief A vector of feature mask. 1 = feature used, 0 = feature not used */
  //std::vector<char, boost::alignment::aligned_allocator<char, 4096>> feature_masks_;
  //std::vector<char, aligned_allocator<char, 4096>> feature_masks_;
  std::vector<char, aligned_allocator<char>> feature_masks_;
  /*! \brief FPGA memory object holding the feature masks */
  //boost::compute::buffer device_feature_masks_;
  cl::Buffer device_feature_masks_;
  /*! \brief Pinned memory object for feature masks */
  //boost::compute::buffer pinned_feature_masks_;
  cl::Buffer pinned_feature_masks_;
  /*! \brief Pointer to pinned memory of feature masks */
  void * ptr_pinned_feature_masks_ = nullptr;
  /*! \brief FPGA memory object holding indices of the leaf being processed */
  //std::unique_ptr<boost::compute::vector<data_size_t>> device_data_indices_;
  std::unique_ptr<cl::vector<data_size_t>> device_data_indices_;
  /*! \brief FPGA memory object holding counters for workgroup coordination */
  //std::unique_ptr<boost::compute::vector<int>> sync_counters_;
  std::unique_ptr<cl::vector<int>> sync_counters_;
  /*! \brief FPGA memory object holding temporary sub-histograms per workgroup */
  //std::unique_ptr<boost::compute::vector<char>> device_subhistograms_;
  std::unique_ptr<cl::vector<char>> device_subhistograms_;
  /*! \brief Host memory object for histogram output (FPGA will write to Host memory directly) */
  //boost::compute::buffer device_histogram_outputs_;
  cl::Buffer device_histogram_outputs_;
  /*! \brief Host memory pointer for histogram outputs */
  void * host_histogram_outputs_;
  /*! \brief OpenCL waitlist object for waiting for data transfer before kernel execution */
  //boost::compute::wait_list kernel_wait_obj_;
  //std::array<cl::Event> kernel_wait_obj_;
  cl::Event kernel_wait_obj_;
  /*! \brief OpenCL waitlist object for reading output histograms after kernel execution */
  //boost::compute::wait_list histograms_wait_obj_;
  //std::array<cl::Event> histograms_wait_obj_;
  cl::Event histograms_wait_obj_;
  /*! \brief Asynchronous waiting object for copying indices */
  //boost::compute::future<void> indices_future_;
  cl::Event indices_future_;
  /*! \brief Asynchronous waiting object for copying gradients */
  //boost::compute::event gradients_future_;
  cl::Event gradients_future_;
  /*! \brief Asynchronous waiting object for copying hessians */
  //boost::compute::event hessians_future_;
  cl::Event hessians_future_;
};

}  // namespace LightGBM
#else

// When FPGA support is not compiled in, quit with an error message

namespace LightGBM {
    
class FPGATreeLearner: public SerialTreeLearner {
public:
  #pragma warning(disable : 4702)
  explicit FPGATreeLearner(const Config* tree_config) : SerialTreeLearner(tree_config) {
    Log::Fatal("FPGA Tree Learner was not enabled in this build.\n"
               "Please recompile with CMake option -DUSE_FPGA=1");
  }
};

}

#endif   // USE_FPGA

#endif   // LightGBM_TREELEARNER_FPGA_TREE_LEARNER_H_

