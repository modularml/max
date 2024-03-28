/*******************************************************************************
 * Copyright (c) 2024, Modular Inc. All rights reserved.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions:
 * https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "max/c/common.h"
#include "max/c/context.h"
#include "max/c/model.h"
#include "max/c/tensor.h"
#include "max/c/tensorflow/config.h"

#include <stdio.h>
#include <stdlib.h>

void logHelper(const char *level, const char *message, const char delimiter) {
  printf("%s: %s%c", level, message, delimiter);
}

void logDebug(const char *message) { logHelper("DEBUG", message, ' '); }

void logInfo(const char *message) { logHelper("INFO", message, '\n'); }

void logError(const char *message) { logHelper("ERROR", message, '\n'); }

// Read file at the given path. On failure abort.
char *readFileOrExit(const char *filepath) {
  FILE *file;
  file = fopen(filepath, "rb");
  if (!file) {
    printf("failed to open %s. Aborting.\n", filepath);
    abort();
  }
  fseek(file, 0, SEEK_END);
  long fileSize = ftell(file);
  rewind(file);

  char *buffer = (char *)malloc(fileSize * sizeof(char));
  fread(buffer, fileSize, 1, file);
  fclose(file);
  return buffer;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: bert <path to bert saved model>");
    return EXIT_FAILURE;
  }

  M_Status *status = M_newStatus();

  M_RuntimeConfig *runtimeConfig = M_newRuntimeConfig();
  M_RuntimeContext *context = M_newRuntimeContext(runtimeConfig, status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  logInfo("Compiling Model");
  M_CompileConfig *compileConfig = M_newCompileConfig();
  const char *modelPath = argv[1];
  M_setModelPath(compileConfig, /*path=*/modelPath);

  M_AsyncCompiledModel *compiledModel =
      M_compileModel(context, &compileConfig, status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  logInfo("Initializing Model");
  M_AsyncModel *model = M_initModel(context, compiledModel, status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  M_TensorNameArray *tensorNames = M_getInputNames(compiledModel, status);
  size_t numInputs = M_getNumModelInputs(compiledModel, status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  logDebug("Model input names:");
  for (size_t i = 0; i < numInputs; i++) {
    const char *tensorName = M_getTensorNameAt(tensorNames, i);
    printf("%s ", tensorName);
  }
  printf("\n");

  // Define the input tensor specs.
  int64_t *inputIdsShape =
      (int64_t *)readFileOrExit("inputs/input_ids_shape.bin");
  M_TensorSpec *inputIdsSpec =
      M_newTensorSpec(inputIdsShape, /*rankSize=*/2, /*dtype=*/M_INT32,
                      /*tensorName=*/"input_ids");
  free(inputIdsShape);

  int64_t *attentionMaskShape =
      (int64_t *)readFileOrExit("inputs/attention_mask_shape.bin");
  M_TensorSpec *attentionMaskSpec =
      M_newTensorSpec(attentionMaskShape, /*rankSize=*/2, /*dtype=*/M_INT32,
                      /*tensorName=*/"attention_mask");
  free(attentionMaskShape);

  int64_t *tokenTypeIdsShape =
      (int64_t *)readFileOrExit("inputs/token_type_ids_shape.bin");
  M_TensorSpec *tokenTypeIdsSpec =
      M_newTensorSpec(tokenTypeIdsShape, /*rankSize=*/2, /*dtype=*/M_INT32,
                      /*tensorName=*/"token_type_ids");
  free(tokenTypeIdsShape);

  // Create the input tensor and borrow it into the model input.
  // Borrowing the input means we don't do any copy and caller is responsible
  // to make sure that the input stays alive till the inference is completed.
  M_AsyncTensorMap *inputToModel = M_newAsyncTensorMap(context);
  int32_t *inputIdsTensor = (int32_t *)readFileOrExit("inputs/input_ids.bin");
  M_borrowTensorInto(inputToModel, inputIdsTensor, inputIdsSpec, status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  int32_t *attentionMaskTensor =
      (int32_t *)readFileOrExit("inputs/attention_mask.bin");
  M_borrowTensorInto(inputToModel, attentionMaskTensor, attentionMaskSpec,
                     status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  int32_t *tokenTypeIdsTensor =
      (int32_t *)readFileOrExit("inputs/token_type_ids.bin");
  M_borrowTensorInto(inputToModel, tokenTypeIdsTensor, tokenTypeIdsSpec,
                     status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  // Run the inference.
  // This function blocks until the inference is complete.
  logInfo("Running Inference");
  M_AsyncTensorMap *outputs =
      M_executeModelSync(context, model, inputToModel, status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  logInfo("Inference successfully completed");
  M_AsyncTensor *logits =
      M_getTensorByNameFrom(outputs,
                            /*tensorName=*/"logits", status);
  if (M_isError(status)) {
    logError(M_getError(status));
    return EXIT_FAILURE;
  }

  size_t numElements = M_getTensorNumElements(logits);
  M_Dtype dtype = M_getTensorType(logits);
  const char *outputFilePath = "outputs.bin";
  FILE *file = fopen(outputFilePath, "wb");
  if (!file) {
    printf("failed to open %s. Aborting.\n", outputFilePath);
    return EXIT_FAILURE;
  }
  fwrite(M_getTensorData(logits), M_sizeOf(dtype), numElements, file);
  fclose(file);

  // Cleanup.
  free(inputIdsTensor);
  free(attentionMaskTensor);
  free(tokenTypeIdsTensor);

  M_freeTensor(logits);
  M_freeTensorSpec(inputIdsSpec);
  M_freeTensorSpec(attentionMaskSpec);
  M_freeTensorSpec(tokenTypeIdsSpec);
  M_freeAsyncTensorMap(inputToModel);
  M_freeAsyncTensorMap(outputs);
  M_freeTensorNameArray(tensorNames);

  M_freeModel(model);
  M_freeCompiledModel(compiledModel);
  M_freeRuntimeContext(context);
  M_freeRuntimeConfig(runtimeConfig);
  M_freeStatus(status);
  return EXIT_SUCCESS;
}
