/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
/* 
const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "yes",
    "no",
};

const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "yes",
    "no",
    "clapping",
    "gun_shot",
    "crying_baby",
    "door_knock",
    "sheila",
    "dog_bark",
};
*/

const char* kCategoryLabels[kCategoryCount] = {
    "silence",
    "unknown",
    "car_horn",
    "coughing",
    "clapping",
    "gun_shot",
    "crying_baby",
    "door_knock",
    "clock_alarm",
    "dog",
    "can_opening",
    "children_playing",
    "chainsaw",
    "crackling_fire",
    "footsteps",
    "engine"
};
//car_horn,coughing,clapping,gun_shot,crying_baby,door_knock,clock_alarm,dog,
//can_opening,children_playing,chainsaw,crackling_fire,footsteps,engine
//}