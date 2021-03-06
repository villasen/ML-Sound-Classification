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

#include "tensorflow/lite/micro/examples/micro_speech/main_functions.h"

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"
#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/recognize_commands.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "tensorflow/lite/micro/examples/micro_speech/micro_features/no_micro_features_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/yes_micro_features_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/doorknock_micro_features_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/gun_shot_micro_features_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/crying_baby_micro_features_data.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/dog_bark_micro_features_data.h"


#define MAX_FEATURE_NUMBER 1960
// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
//tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024; // highest 
                                            // available is 70*1024
uint8_t tensor_arena[kTensorArenaSize];
//uint8_t feature_buffer[kFeatureElementCount];
//uint8_t* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;


error_reporter->Report("\n*****Starting Sound Recognition Program for DISCO746*****\n");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_tiny_conv_micro_features_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::ops::micro::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  
  // Depthwise op layer
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D(), 1, 3);

/*
  micro_mutable_op_resolver.AddBuiltin( 
      tflite::BuiltinOperator_QUANTIZE,
      tflite::ops::micro::Register_QUANTIZE(), 1, 2);
*/

 // fully connected op layer
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED(), 1, 4);


  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_MAX_POOL_2D,
      tflite::ops::micro::Register_MAX_POOL_2D(), 1, 1);

  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_CONV_2D,
      tflite::ops::micro::Register_CONV_2D(), 1, 1);

// Reshape operator
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_RESHAPE ,
      tflite::ops::micro::Register_RESHAPE());     
  // Softmax op layer
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                       tflite::ops::micro::Register_SOFTMAX(), 1, 2);

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_mutable_op_resolver,
                                    tensor_arena, kTensorArenaSize,
                                    error_reporter);

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter.AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report( "AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter.input(0);

  error_reporter->Report("model dim size=%d", model_input->dims->size);
  error_reporter->Report("model dim data0=%d", model_input->dims->data[0]);
  error_reporter->Report("model dim data1=%d", model_input->dims->data[1]);
  error_reporter->Report("model dim data2=%d", model_input->dims->data[2]);
  error_reporter->Report("model type=%d", model_input->type);

 error_reporter->Report("Wait... Thinking...", model_input->type);
/*
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteUInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }
*/

// Copy a spectrogram created from a .wav audio file 
  // into the memory area used for the input.
//  const uint8_t* features_data = g_yes_micro_f2e59fea_nohash_1_data;

//** door knock 
//  const uint8_t* features_data = g_door_knock_a_1_26188_a_data;
//  const uint8_t* features_data = g_door_knock_a_1_52290_a_data;
//  const uint8_t* features_data = g_door_knock_a_1_81001_a_data;
// const uint8_t* features_data =  g_door_knock_a_1_81001_b_data;
//const uint8_t* features_data =  g_door_knock_a_1_82817_a_data;
//const uint8_t* features_data =  g_door_knock_a_1_101336_a_data;
//const uint8_t* features_data =  g_door_knock_a_1_103995_a_data;
//const uint8_t* features_data =  g_door_knock_a_1_103999_a_data;
//const uint8_t* features_data =  g_door_knock_a_2_114254_a_data;
//const uint8_t* features_data = g_door_knock_a_2_118624_a_data;
//const uint8_t* features_data = g_door_knock_a_2_118625_a_data;

//** gun shot
//const uint8_t* features_data =  g_gun_shot_7060_6_0_0_data;
//const uint8_t* features_data =  g_gun_shot_7060_6_1_0_data;
//const uint8_t* features_data =  g_gun_shot_7060_6_2_0_data;
//const uint8_t* features_data =  g_gun_shot_7061_6_0_0_data;
//const uint8_t* features_data =  g_gun_shot_7062_6_0_0_data;
//const uint8_t* features_data =  g_gun_shot_7063_6_0_0_data;
//const uint8_t* features_data =  g_gun_shot_7064_6_0_0_data;
//const uint8_t* features_data =  g_gun_shot_7064_6_1_0_data;
//const uint8_t* features_data =  g_gun_shot_7064_6_2_0_data;
//const uint8_t* features_data = g_gun_shot_7064_6_4_0_data;

const uint8_t *sounds_array[3]
{
    g_crying_baby_a_1_187207_a_data,
    g_crying_baby_a_2_50665_a_data,
    g_crying_baby_a_5_198411_a_data
 };

//const uint8_t**  features_data = sounds_array;
//** crying baby
const uint8_t* features_data =  g_crying_baby_a_1_187207_a_data;
//const uint8_t* features_data =    g_crying_baby_a_2_50665_a_data;
//const uint8_t* features_data =  g_crying_baby_a_5_198411_a_data;
//const uint8_t* features_data =  g_crying_baby_a_5_198411_a_data;
//const uint8_t* features_data =  g_crying_baby_b_2_50665_a_data;
//const uint8_t* features_data =  g_crying_baby_b_2_80482_a_data;
//const uint8_t* features_data =  g_crying_baby_b_5_198411_d_data;
//const uint8_t* features_data =  g_crying_baby_c_3_152007_e_data;
//const uint8_t* features_data =  g_crying_baby_c_5_198411_a_data;
//const uint8_t* features_data =  g_crying_baby_c_5_198411_b_data;
//const uint8_t* features_data =  g_crying_baby_d_4_167077_b_data;
//const uint8_t* features_data =  g_crying_baby_e_5_198411_d_data;



//** dog bark
//const uint8_t* features_data =  g_dog_bark_22973_3_0_0_data;
//const uint8_t* features_data = g_dog_bark_26256_3_7_36_data;
//const uint8_t* features_data3 = g_dog_bark_33696_3_4_0_data;
//const uint8_t* features_data4 = g_dog_bark_52077_3_0_13_data;
//const uint8_t* features_data5 = g_dog_bark_66587_3_1_0_data;
//const uint8_t* features_data6 = g_dog_bark_76640_3_0_0_data;
//const uint8_t* features_data7 = g_dog_bark_81799_3_1_0_data;
//const uint8_t* features_data8 = g_dog_bark_118101_3_0_0_data;
//const uint8_t* features_data9 = g_dog_bark_118962_3_0_0_data;
//const uint8_t* features_data10 = g_dog_bark_175915_3_0_1_data;
//const uint8_t* features_data11 = g_dog_bark_183989_3_1_18_data;


//const uint8_t* features_data1 =  g_dog_bark_22973_3_0_0_data;
//const uint8_t* features_data2 = g_dog_bark_26256_3_7_36_data;
//const uint8_t* features_data3 = g_dog_bark_33696_3_4_0_data;
//const uint8_t* features_data4 = g_dog_bark_52077_3_0_13_data;
//const uint8_t* features_data5 = g_dog_bark_66587_3_1_0_data;
//const uint8_t* features_data6 = g_dog_bark_76640_3_0_0_data;
//const uint8_t* features_data7 = g_dog_bark_81799_3_1_0_data;
//const uint8_t* features_data8 = g_dog_bark_118101_3_0_0_data;
//const uint8_t* features_data9 = g_dog_bark_118962_3_0_0_data;
//const uint8_t* features_data10 = g_dog_bark_175915_3_0_1_data;
//const uint8_t* features_data11 = g_dog_bark_183989_3_1_18_data;

//for (k ; k < 3 ; k++ )
//{

  // error_reporter->Report("Testing loop #%d", k);
   
  // const uint8_t* features_data =  dog_bark_feature_data;   
  //g_dog_bark_183989_3_1_18_data;
  //  temp[k] = dog_bark_feature_data[k];

   // features_data =  ptr_dog_bark_feature_data+0;
   // error_reporter->Report("Features_data[0] = %d", features_data[0]);
  //const uint8_t* features_data = g_no_micro_f9643d42_nohash_4_data;




    for (int i = 0; i < model_input->bytes; ++i) {
       model_input->data.uint8[i] = features_data[i]; 
    //model_input->data.uint8[i] = **features_data;
   //   model_input->data.uint8[i] = sounds_array[1];
        
 
    }
    
  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter.output(0);
  error_reporter->Report("output: %d", output->data.uint8[0]);

  
  
 /* 
  // There are four possible classes in the output, each with a score.
  const int kSilenceIndex = 0;
  const int kUnknownIndex = 1;
  const int kYesIndex = 2;
  const int kNoIndex = 3;
  const int kClappingIndex = 4; 
  const int kGunShotIndex = 5;
  const int kCryingBabyIndex = 6;
  const int kDoorKnockIndex = 7;
  const int kSheilaIndex = 8;
  const int kDogBarkIndex = 9;

  // Make sure that the expected "Yes" score is higher than the other classes.
 
  uint8_t silence_score = output->data.uint8[kSilenceIndex];
  uint8_t unknown_score = output->data.uint8[kUnknownIndex];
  uint8_t yes_score = output->data.uint8[kYesIndex];
  uint8_t no_score = output->data.uint8[kNoIndex];
  uint8_t clapping_score = output->data.uint8[kClappingIndex];
  uint8_t gunshot_score = output->data.uint8[kGunShotIndex];
  uint8_t crying_baby_score = output->data.uint8[kCryingBabyIndex]; 
  uint8_t door_knock_score = output->data.uint8[kDoorKnockIndex]; 
  uint8_t sheila_score = output->data.uint8[kSheilaIndex];
  uint8_t dog_bark_score = output->data.uint8[kDogBarkIndex];
*/

  uint8_t silence_score = output->data.uint8[kSilenceIndex];   //1
  uint8_t unknown_score = output->data.uint8[kUnknownIndex];
  uint8_t car_horn_score = output->data.uint8[kCarHornIndex];
  uint8_t coughing_score = output->data.uint8[kCoughingIndex];
  uint8_t clapping_score = output->data.uint8[kClappingIndex];  //5
  uint8_t gun_shot_score = output->data.uint8[kGunShotIndex];
  uint8_t crying_baby_score = output->data.uint8[kCryingBabyIndex]; 
  uint8_t door_knock_score = output->data.uint8[kDoorKnockIndex]; 
  uint8_t clock_alarm_score = output->data.uint8[kClockAlarmIndex];
  uint8_t dog_score = output->data.uint8[kDogIndex];   //10
  uint8_t can_opening_score = output->data.uint8[kCanOpeningIndex];
  uint8_t children_playing_score = output->data.uint8[kChildrenPlayingIndex];
  uint8_t chainsaw_score = output->data.uint8[kChainSawIndex];
  uint8_t crackling_fire_score = output->data.uint8[kCracklingFireIndex];
  uint8_t footsteps_score = output->data.uint8[kFootStepsIndex];  //15
  uint8_t engine_score = output->data.uint8[kEngineIndex];

//car_horn,coughing,clapping,gun_shot,crying_baby,door_knock,clock_alarm,dog,
//can_opening,children_playing,chainsaw,crackling_fire,footsteps,engine
//}


  error_reporter->Report("Softmax: silence=%d, unknown=%d, car horn=%d, coughing=%d,  \
  clapping=%d, gunshot=%d, crying_baby=%d, door_knock=%d, clock alarm=%d, dog=%d  \
  can opening=%d, children playing=%d, chain saw=%d, crackling fire=%d, foot steps=%d \
  engine=%d" ,silence_score, unknown_score, car_horn_score, coughing_score, clapping_score, 
  gun_shot_score, crying_baby_score, door_knock_score,
  clock_alarm_score , dog_score, can_opening_score, children_playing_score,
  chainsaw_score, crackling_fire_score, footsteps_score, engine_score ); 

//car_horn,coughing,clapping,gun_shot,crying_baby,door_knock,clock_alarm,dog,
//can_opening,children_playing,chainsaw,crackling_fire,footsteps,engin
//}

 
  error_reporter->Report("\n*****End of Sound Recognition Classifier for DISCO746*****");

  // Determine whether a command was recognized based on the output of inference
 // const char* found_command = nullptr;
 // uint8_t score = 0;
//  bool is_new_command = false;
//  TfLiteStatus process_status = recognizer->ProcessLatestResults(
 //     output, current_time, &found_command, &score, &is_new_command);
 // if (process_status != kTfLiteOk) {
 //   TF_LITE_REPORT_ERROR(error_reporter,
 //                        "RecognizeCommands::ProcessLatestResults() failed");
  //  return;
 // }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  
 // RespondToCommand(error_reporter, current_time, found_command, score,
 //                  is_new_command);
  
  
  //RespondToCommands(error_reporter, found_command, score,
  //                 is_new_command);


}
