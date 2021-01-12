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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {

class XStatVisitor {
 public:
  XStatVisitor(const XPlane* plane, const XStat* stat)
      : stat_(stat),
        metadata_(&gtl::FindWithDefault(plane->stat_metadata(),
                                        stat_->metadata_id(),
                                        XStatMetadata::default_instance())) {}

  int64 Id() const { return stat_->metadata_id(); }

  absl::string_view Name() const { return metadata_->name(); }

  absl::string_view Description() const { return metadata_->description(); }

  XStat::ValueCase ValueCase() const { return stat_->value_case(); }

  int64 IntValue() const { return stat_->int64_value(); }

  uint64 UintValue() const { return stat_->uint64_value(); }

  double DoubleValue() const { return stat_->double_value(); }

  absl::string_view StrValue() const { return stat_->str_value(); }

  const XStat& RawStat() const { return *stat_; }

 private:
  const XStat* stat_;
  const XStatMetadata* metadata_;
};

class XEventVisitor {
 public:
  XEventVisitor(const XPlane* plane, const XLine* line, const XEvent* event)
      : plane_(plane),
        line_(line),
        event_(event),
        metadata_(&gtl::FindWithDefault(plane_->event_metadata(),
                                        event_->metadata_id(),
                                        XEventMetadata::default_instance())) {}

  int64 Id() const { return event_->metadata_id(); }

  absl::string_view Name() const { return metadata_->name(); }

  absl::string_view DisplayName() const {
    return !metadata_->display_name().empty() ? metadata_->display_name()
                                              : metadata_->name();
  }

  absl::string_view Metadata() const { return metadata_->metadata(); }

  double OffsetNs() const { return PicosToNanos(event_->offset_ps()); }

  int64 OffsetPs() const { return event_->offset_ps(); }

  int64 LineTimestampNs() const { return line_->timestamp_ns(); }

  double TimestampNs() const { return line_->timestamp_ns() + OffsetNs(); }

  int64 TimestampPs() const {
    return NanosToPicos(line_->timestamp_ns()) + event_->offset_ps();
  }

  double DurationNs() const { return PicosToNanos(event_->duration_ps()); }

  int64 DurationPs() const { return event_->duration_ps(); }

  int64 EndOffsetPs() const {
    return event_->offset_ps() + event_->duration_ps();
  }

  int64 NumOccurrences() const { return event_->num_occurrences(); }

  template <typename ForEachStatFunc>
  void ForEachStat(ForEachStatFunc&& for_each_stat) const {
    for (const XStat& stat : event_->stats()) {
      for_each_stat(XStatVisitor(plane_, &stat));
    }
  }

  bool operator<(const XEventVisitor& other) const {
    return GetTimespan() < other.GetTimespan();
  }

 private:
  Timespan GetTimespan() const { return Timespan(TimestampPs(), DurationPs()); }

  const XPlane* plane_;
  const XLine* line_;
  const XEvent* event_;
  const XEventMetadata* metadata_;
};

class XLineVisitor {
 public:
  XLineVisitor(const XPlane* plane, const XLine* line)
      : plane_(plane), line_(line) {}

  int64 Id() const { return line_->id(); }

  int64 DisplayId() const {
    return line_->display_id() ? line_->display_id() : line_->id();
  }

  absl::string_view Name() const { return line_->name(); }

  absl::string_view DisplayName() const {
    return !line_->display_name().empty() ? line_->display_name()
                                          : line_->name();
  }

  double TimestampNs() const { return line_->timestamp_ns(); }

  int64 DurationPs() const { return line_->duration_ps(); }

  size_t NumEvents() const { return line_->events_size(); }

  template <typename ForEachEventFunc>
  void ForEachEvent(ForEachEventFunc&& for_each_event) const {
    for (const XEvent& event : line_->events()) {
      for_each_event(XEventVisitor(plane_, line_, &event));
    }
  }

 private:
  const XPlane* plane_;
  const XLine* line_;
};

class XPlaneVisitor {
 public:
  explicit XPlaneVisitor(const XPlane* plane) : plane_(plane) {}

  int64 Id() const { return plane_->id(); }

  absl::string_view Name() const { return plane_->name(); }

  size_t NumLines() const { return plane_->lines_size(); }

  template <typename ForEachLineFunc>
  void ForEachLine(ForEachLineFunc&& for_each_line) const {
    for (const XLine& line : plane_->lines()) {
      for_each_line(XLineVisitor(plane_, &line));
    }
  }

 private:
  const XPlane* plane_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_XPLANE_VISITOR_H_
