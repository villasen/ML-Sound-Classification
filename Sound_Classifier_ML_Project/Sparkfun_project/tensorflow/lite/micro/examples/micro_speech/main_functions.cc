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
constexpr int kTensorArenaSize = 7 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

error_reporter->Report("\n*****Starting Sound Recognition Program for Sparkfun Edge*****\n");
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_tiny_conv_micro_features_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
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


  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_mutable_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_FULLY_CONNECTED,
      tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_mutable_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                                       tflite::ops::micro::Register_SOFTMAX());


  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_mutable_op_resolver,
                                    tensor_arena, kTensorArenaSize,
                                    error_reporter);

 
  TfLiteStatus allocate_status = interpreter.AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  //model_input = interpreter->input(0);
  model_input = interpreter.input(0);

  error_reporter->Report("model dim size=%d", model_input->dims->size);
  error_reporter->Report("model dim data0=%d", model_input->dims->data[0]);
  error_reporter->Report("model dim data1=%d", model_input->dims->data[1]);
  error_reporter->Report("model dim data2=%d", model_input->dims->data[2]);
  error_reporter->Report("model type=%d", model_input->type);



  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->type != kTfLiteUInt8)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return;
  }

// Copy a spectrogram created from a .wav audio file 
  // into the memory area used for the input.



  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
 // static FeatureProvider static_feature_provider(kFeatureElementCount,
 //                                                model_input->data.uint8);
  //feature_provider = &static_feature_provider;

 // static RecognizeCommands static_recognizer(error_reporter);
 // recognizer = &static_recognizer;

 // previous_time = 0;
//}

  // Fetch the spectrogram for the current time.
  //const int32_t current_time = LatestAudioTimestamp();
  //int how_many_new_slices = 0;
 // TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
  //    error_reporter, previous_time, current_time, &how_many_new_slices);
  //if (feature_status != kTfLiteOk) {
 //   error_reporter->Report("Feature generation failed");
  //  return;
  //}
 // previous_time = current_time;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  //if (how_many_new_slices == 0) {
  //  return;
 // }

 
 // Copy a spectrogram created from a .wav audio file 
  // into the memory area used for the input.
 // const uint8_t* features_data1 = g_yes_micro_f2e59fea_nohash_1_data;
 // const uint8_t* features_data1 = g_dog_bark_183989_3_1_18_data; 
  //const uint8_t* features_data1 = g_crying_baby_a_5_198411_a_data;

  
  const uint8_t *sounds_array[6]
  {
    g_crying_baby_a_1_187207_a_data,
    g_crying_baby_a_2_50665_a_data,
    g_crying_baby_a_5_198411_a_data,
    g_dog_bark_81799_3_1_0_data,
    g_door_knock_a_1_26188_a_data,
    g_yes_micro_f2e59fea_nohash_1_data

  };


    const uint8_t**  pfeatures_data;
    pfeatures_data = sounds_array;
    unsigned char spectogram_data;
    //spectogram_data = 
for (int i = 0; i < BABY_TOTAL_FILE_NUMBER ; i++)
{ 
    error_reporter->Report("Test Number is %d", i);
    error_reporter->Report("*****************");
    error_reporter->Report("Testing input data crying baby");


  for (int j = 0; j < model_input->bytes; j++) {
       model_input->data.uint8[j] =  g_crying_baby_sounds_arrays[i][j];
    //   model_input->data.uint8[j] =  g_dog_bark_sounds_arrays[i][j];
    //   model_input->data.uint8[j] =  g_door_knock_sounds_arrays[i][j];
    //   model_input->data.uint8[j] =  g_gun_shot_sounds_arrays[i][j];   
    }
 
 
  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter.Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  //TfLiteTensor* output = interpreter->output(0);
  TfLiteTensor* output = interpreter.output(0);
  error_reporter->Report("output: %d", output->data.uint8[0]);

  
  // There are four possible classes in the output, each with a score.
  const int kSilenceIndex = 0;
  const int kUnknownIndex = 1;
  const int kcarhornIndex = 2;
  const int kCoughingIndex = 3;
  const int kClappingIndex = 4; 
  const int kGunShotIndex = 5;
  const int kCryingBabyIndex = 6;
  const int kDoorKnockIndex = 7;
  const int kSheilaIndex = 8;
  const int kDogBarkIndex = 9;
  
  
  // Make sure that the expected "Yes" score is higher than the other classes.
  uint8_t silence_score = output->data.uint8[kSilenceIndex];
  uint8_t unknown_score = output->data.uint8[kUnknownIndex];
  uint8_t car_horn_score = output->data.uint8[kcarhornIndex];
  uint8_t coughing_score = output->data.uint8[kCoughingIndex];
  uint8_t clapping_score = output->data.uint8[kClappingIndex];
  uint8_t gunshot_score = output->data.uint8[kGunShotIndex];
  uint8_t crying_baby_score = output->data.uint8[kCryingBabyIndex]; 
  uint8_t door_knock_score = output->data.uint8[kDoorKnockIndex]; 
  uint8_t sheila_score = output->data.uint8[kSheilaIndex];
  uint8_t dog_bark_score = output->data.uint8[kDogBarkIndex];

 // error_reporter->Report("Softmax: silence=%d, unknown=%d, yes=%d, no=%d", silence_score, unknown_score, yes_score, no_score); 
  //"car_horn,coughing,clapping,gun_shot,crying_baby,door_knock,clock_alarm,dog"
  error_reporter->Report("Softmax: silence=%d, unknown=%d, car horn=%d, coughing=%d,  \
  clapping=%d, gunshot=%d, crying_baby=%d, door_knock=%d, clock alarm=%d, dog=%d" 
  ,silence_score, unknown_score, kcarhornIndex, coughing_score, clapping_score, 
  gunshot_score, crying_baby_score, door_knock_score,
  sheila_score, dog_bark_score);  

}  

  error_reporter->Report("\n*****Martin is classifying sounds*****");
  error_reporter->Report("\n*****End of Sound Recognition Classifier for Sparfun Edge Program*****");

  // Determine whether a command was recognized based on the output of inference
 // const char* found_command = nullptr;
 // uint8_t score = 0;
 // bool is_new_command = false;
//  TfLiteStatus process_status = recognizer->ProcessLatestResults(
 //     output, current_time, &found_command, &score, &is_new_command);
 // if (process_status != kTfLiteOk) {
 //   error_reporter->Report("RecognizeCommands::ProcessLatestResults() failed");
 //   return;
 // }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
 // RespondToCommand(error_reporter, current_time, found_command, score,
 //                  is_new_command);
}
