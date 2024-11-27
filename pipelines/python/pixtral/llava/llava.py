# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from dataclasses import dataclass

from max.graph import TensorValue, ops
from nn.layer import Layer
from pixtral.vision_encoder.vision_encoder import VisionEncoder

from .llava_decoder import Transformer
from .llava_projector import LlavaMultiModalConnector


@dataclass
class LlavaConditionalGeneration(Layer):
    """The LLAVA model which consists of a vision encoder and a language model.

    image_token_index: a specific token index used to denote images
    """

    vision_encoder: VisionEncoder
    multi_modal_projector: LlavaMultiModalConnector
    language_model: Transformer
    vocab_size: int
    pad_token_id: int = -1
    image_token_index: int = 10
    vision_feature_layer: int = -1
    vision_feature_select_strategy: str = "full"
    image_seq_length: int = 1

    # TODO: change pixel_values type to List[TensorValue] to support multiple images.
    def __call__(
        self,
        input_ids: TensorValue,  # Shape (batch_size, sequence_length). Indices of input sequence tokens in the vocabulary. Indices can be obtained from language model tokenizer.
        pixel_values: TensorValue,  # Shape list of length n_images of (height, width, num_channels).
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> TensorValue:
        """
        Args:
            input_ids (`TensorValue` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
                Image tokens are assumed to have index self.image_token_index.
                The maximum number of image tokens in one sequence (prompt) =
                    (input_ids == self.config.image_token_index).sum(1).max())
                Padding will be ignored by default should you provide it.
                input_ids maps each input to a sequence of tokens representing it.
                It maps image tokens to a sequence of patches of length n_patches
                that all have value = self.image_token_index
            pixel_values (`TensorValue` of shape `(batch_size, image_size, image_size, num_channels)):
                The tensors corresponding to the input images. Pixel values can be obtained using ImageProcessor
        """

        # TODO: take this as input rather than calculating it here?
        # TODO: if the input is a list, change vision_encoder input to pixel_values
        # Obtains image last.  Output shape = (num_images=batch_size, num_patches_in_image, vision_encoder_hidden_dim)
        # Apply multimodal projection to  hidden states from the vision encoder.
        image_embeds = self.multi_modal_projector(
            self.vision_encoder(
                [
                    pixel_values,
                ]
            )
        )  # image_embeds shape = (num_images, num_patches_in_image, language_model_hidden_dim)
        # inputs_embeds shape
        inputs_embeds = self.language_model.embedding(input_ids)
        print("inputs_embeds shape = ", inputs_embeds.shape)
        # Replace image place-holders in inputs_embeds with image embeddings.
        special_image_mask = ops.broadcast_to(
            ops.unsqueeze((input_ids == self.image_token_index), -1),
            inputs_embeds.shape,
        )
        image_embeds = ops.cast(image_embeds, inputs_embeds.dtype)
        inputs_embeds = ops.masked_scatter(
            inputs_embeds, special_image_mask, image_embeds
        )
        print("inputs_embeds shape after scatter = ", inputs_embeds.shape)

        logits = self.language_model(inputs_embeds, kv_cache_inputs, **kwargs)
        return logits
        # Expanding inputs for image tokens in LLaVa should be done in processing
        # So, the input to Llava should be:
        # input_ids which have an index (placeholder) for each token (patch) in
        # each image rather than one token for the image.
        # max_tokens_per_image = (
        #     (input_ids == self.image_token_index).sum(1).max()
        # )
        # assert(max_tokens_per_image < self.image_seq_length)

        # n_image_embeds = n_images * n_patches_per_image
        # image_embeds.shape = [Dim(1), Dim(475), Dim(5120)]
        # n_image_embeds = image_embeds.shape[0] * image_embeds.shape[1]
        # print("image_embeds.shape", "n_image_features", image_embeds.shape, n_image_embeds)

        # n_image_tokens = ops.sum(ops.sum(ops.cast((input_ids == self.image_token_index), DType.int32)))[0]

        # print("n_image_tokens", n_image_tokens)

        # if n_image_tokens != ops.constant(int(n_image_embeds), DType.int32):
        #     raise ValueError(
        #         "Image sfeatures and image tokens do not match: tokens:"
        #         f" {n_image_tokens}, features {n_image_embeds}"
        #     )
