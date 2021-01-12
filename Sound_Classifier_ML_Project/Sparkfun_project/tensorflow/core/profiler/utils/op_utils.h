/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_

#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/op_metrics_db_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"

namespace tensorflow {
namespace profiler {

class HostOpMetricsDbBuilder : public OpMetricsDbBuilder {
 public:
  explicit HostOpMetricsDbBuilder(OpMetricsDb* db) : OpMetricsDbBuilder(db) {}

  // A function that will be called when the end of an OP is
  // observed on a trace, where:
  //   name = the OP name.
  //   category = the OP category.
  //   time_ps = the total execution time of the OP in picoseconds, including
  //             the execution time of its children.
  //   children_time_ps = the execution time of the children of this OP in
  //                      picoseconds
  void EnterOp(absl::string_view name, absl::string_view category,
               uint64 time_ps, uint64 children_time_ps);

  // Updates total_host_infeed_enq_duration_ps_ and
  // total_host_infeed_enq_duration_ps_.
  void UpdateHostInfeedEnqInfo(uint64 duration_ps,
                               uint64 start_timestamp_ps_diff);
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_OP_UTILS_H_
