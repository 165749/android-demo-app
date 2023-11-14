#include <android/log.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <unistd.h>
#include <vector>
#define ALOGI(...)                                                             \
  __android_log_print(ANDROID_LOG_INFO, "PyTorchNativeApp", __VA_ARGS__)
#define ALOGE(...)                                                             \
  __android_log_print(ANDROID_LOG_ERROR, "PyTorchNativeApp", __VA_ARGS__)

#include "jni.h"

#define USE_KINETO

#include <torch/script.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
#include <torch/csrc/jit/mobile/profiler_edge.h>
#include <torch/csrc/profiler/events.h>
#include <c10/mobile/CPUCachingAllocator.h>

namespace pytorch_nativeapp {

template <typename T> void log(const char *m, T t) {
    std::ostringstream os;
    os << t << std::endl;
    ALOGI("%s %s", m, os.str().c_str());
}

namespace {

std::vector<std::string>
split(char separator, const std::string& string, bool ignore_empty = true) {
    std::vector<std::string> pieces;
    std::stringstream ss(string);
    std::string item;
    while (getline(ss, item, separator)) {
        if (!ignore_empty || !item.empty()) {
            pieces.push_back(std::move(item));
        }
    }
    return pieces;
}

std::vector<c10::IValue> create_inputs() {

    std::string FLAGS_input_dims = "1,3,224,224";
    std::string FLAGS_input_type = "float";
    std::string FLAGS_input_memory_format = "contiguous_format";
    // std::string FLAGS_input_memory_format = "channels_last"; // for testing

    CAFFE_ENFORCE_GE(FLAGS_input_dims.size(), 0, "Input dims must be specified.");
    CAFFE_ENFORCE_GE(FLAGS_input_type.size(), 0, "Input type must be specified.");

    std::vector<std::string> input_dims_list = split(';', FLAGS_input_dims);
    std::vector<std::string> input_type_list = split(';', FLAGS_input_type);
    std::vector<std::string> input_memory_format_list =
            split(';', FLAGS_input_memory_format);

    CAFFE_ENFORCE_EQ(
            input_dims_list.size(),
            input_type_list.size(),
            "Input dims and type should have the same number of items.");
    CAFFE_ENFORCE_EQ(
            input_dims_list.size(),
            input_memory_format_list.size(),
            "Input dims and format should have the same number of items.");

    std::vector<c10::IValue> inputs;
    for (size_t i = 0; i < input_dims_list.size(); ++i) {
        auto input_dims_str = split(',', input_dims_list[i]);
        std::vector<int64_t> input_dims;
        for (const auto& s : input_dims_str) {
            input_dims.push_back(c10::stoi(s));
        }

        at::ScalarType input_type;
        if (input_type_list[i] == "float") {
            input_type = at::ScalarType::Float;
        } else if (input_type_list[i] == "uint8_t") {
            input_type = at::ScalarType::Byte;
        } else if (input_type_list[i] == "int64") {
            input_type = at::ScalarType::Long;
        } else {
            CAFFE_THROW("Unsupported input type: ", input_type_list[i]);
        }

        at::MemoryFormat input_memory_format;
        if (input_memory_format_list[i] == "channels_last") {
            if (input_dims.size() != 4u) {
                CAFFE_THROW(
                        "channels_last memory format only available on 4D tensors!");
            }
            input_memory_format = at::MemoryFormat::ChannelsLast;
        } else if (input_memory_format_list[i] == "contiguous_format") {
            input_memory_format = at::MemoryFormat::Contiguous;
        } else {
            CAFFE_THROW(
                    "Unsupported input memory format: ", input_memory_format_list[i]);
        }

        inputs.push_back(
                torch::ones(
                        input_dims,
                        at::TensorOptions(input_type).
                                memory_format(input_memory_format)));
    }

    return inputs;
}

template<class T>
class Runner {
public:
    virtual ~Runner() = default;
    virtual c10::IValue run(
            T& module,
            const std::vector<c10::IValue>& inputs,
            std::vector<float>& times) {
        return module.forward(inputs);
    }
};

template<class T>
class vkRunner final : public Runner<T> {
public:
    virtual ~vkRunner() = default;
    virtual c10::IValue run(
            T& module,
            const std::vector<c10::IValue>& inputs,
            std::vector<float>& times) override {
        if (!module.attr("requires_backend_transfers", at::IValue(true)).toBool()) {
            // No need to transfer input/output backends
            CAFFE_THROW("No need to transfer input/output backends!");
        }

        if (inputs_.size() == 0) {
            ALOGI("Upload the input tensor(s) to GPU memory.");
            // Upload the input tensor(s) to GPU memory.
            inputs_.clear();
            inputs_.reserve(inputs.size());
            for (const auto& input : inputs) {
                if (input.isTensor()) {
                    inputs_.emplace_back(at::rand(input.toTensor().sizes()).vulkan());
                }
                else if (input.isTensorList()) {
                    const c10::List<at::Tensor> input_as_list = input.toTensorList();
                    c10::List<at::Tensor> input_vk_list;
                    input_vk_list.reserve(input_as_list.size());
                    for (int i=0; i < input_as_list.size(); ++i) {
                        const at::Tensor element = input_as_list.get(i);
                        input_vk_list.emplace_back(at::rand(element.sizes()).vulkan());
                    }
                    inputs_.emplace_back(c10::IValue(input_vk_list));
                }
                else {
                    CAFFE_THROW("Inputs must only contain IValues of type c10::Tensor or c10::TensorList!");
                }
            }
        }


        // Run, and download the output tensor to system memory.
        c10::IValue output = module.forward(inputs_);

        ALOGI("after forward %lld", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());

        if (output.isTensor()) {
            return output.toTensor().cpu();
        }
        else if (output.isTensorList()) {
            return output.toTensorList().get(0).cpu();
        }
        else if (output.isList()) {
            return output.toList().get(0).toTensor().cpu();
        }
        else if (output.isTuple()) {
            return output.toTuple()->elements()[0].toTensor().cpu();
        }
        else {
            CAFFE_THROW("Outputs must only be either c10::Tensor or c10::TensorList!");
        };
    }

private:
    std::vector<c10::IValue> inputs_;
};

} // namespace

static void loadAndForwardModel(JNIEnv *env, jclass, jstring jModelPath) {
    bool FLAGS_vulkan = false;
    bool FLAGS_use_caching_allocator = true;

    const char *modelPath = env->GetStringUTFChars(jModelPath, 0);
    assert(modelPath);

    std::vector<c10::IValue> inputs = create_inputs();

    ALOGI("%s", modelPath);
    torch::jit::mobile::Module module = torch::jit::_load_for_mobile(modelPath);

    using ModuleType = torch::jit::mobile::Module;
    const auto runner = FLAGS_vulkan ? std::make_unique<vkRunner<ModuleType>>()
    : std::make_unique<Runner<ModuleType>>();

    c10::CPUCachingAllocator caching_allocator;
    c10::optional<c10::WithCPUCachingAllocatorGuard> caching_allocator_guard;
    if (FLAGS_use_caching_allocator) {
        caching_allocator_guard.emplace(&caching_allocator);
    }
    ALOGI("Starting benchmark.");
    ALOGI("Running warmup runs.");

    bool FLAGS_full_profile = false;
    int FLAGS_iter = 10;
    std::vector<float> times;

    // std::string trace_file_name = "/data/data/org.pytorch.nativeapp/files/trace.txt";
    // torch::jit::mobile::KinetoEdgeCPUProfiler profiler(
    //         module,
    //         trace_file_name,
    //         FLAGS_full_profile, // record input_shapes
    //         FLAGS_full_profile, // profile memory
    //         true, // record callstack
    //         FLAGS_full_profile, // record flops
    //         FLAGS_full_profile, // record module hierarchy
    //         {}, // performance events
    //         false); // adjust_vulkan_timestamps
    for (int i = 0; i < FLAGS_iter; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        ALOGI("start %lld", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        auto output = runner->run(module, inputs, times);
        ALOGI("end %lld", std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        times.push_back(duration.count());
    }
    for (const auto & dur: times) {
        ALOGI("%lf", dur);
    }
    // profiler.disableProfiler();
    // const auto & result = profiler.getProfilerResult();
    //
    // ALOGI("%ld", result->trace_start_us());
    // for (const auto & event: result->events()) {
    //     ALOGI("%s", event.name().c_str());
    //     ALOGI("%ld", event.startUs());
    //     ALOGI("%ld", event.durationUs());
    // }

    env->ReleaseStringUTFChars(jModelPath, modelPath);
}

} // namespace pytorch_nativeapp

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("org/pytorch/nativeapp/NativeClient$NativePeer");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"loadAndForwardModel", "(Ljava/lang/String;)V",
       (void *)pytorch_nativeapp::loadAndForwardModel},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
