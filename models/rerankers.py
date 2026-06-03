import os
import h5py
from qwen_vl_utils import fetch_image
from transformers import BatchFeature
import faiss


from .compression import *
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.utils import (
    is_torchdynamo_compiling,
    can_return_tuple,
    auto_docstring,
)


class QwenReranker(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, config, compression_config):
        super().__init__(config)

        print("\nCustom Class built by Bahey\n")
        self.compression_config = compression_config

        self.apply_pq = self.compression_config["product_quantization"]["apply"]
        self.apply_sampling = self.compression_config["sampling"]["apply"]
        self.apply_pooling = self.compression_config["pooling"]["apply"]
        self.apply_grouping = self.compression_config["grouping"]["apply"]
        self.apply_clustering = self.compression_config["clustering"]["apply"]
        self.apply_asym = self.compression_config["asymmetry"]["apply"]
        self.save_tokens = self.compression_config["save_tokens"]["apply"]

        self.modes = {
            "none": not (
                self.apply_sampling
                or self.apply_pooling
                or self.apply_grouping
                or self.apply_clustering
                or self.apply_pq
            ),
            "sampling": self.apply_sampling,
            "pooling": self.apply_pooling,
            "grouping": self.apply_grouping,
            "clustering": self.apply_clustering,
            "pq": self.apply_pq,
        }

        if self.apply_pq:
            index_path = self.compression_config["product_quantization"][
                "index_file_path"
            ]
            self.faiss_index = faiss.read_index(index_path)

        if self.save_tokens:
            self.sample_idx = 0
            self.save_dir = self.compression_config["save_tokens"]["save_dir"]

    def process_image(self, image, processor, pq=False):
        image = fetch_image({"image": image})
        image_inputs = processor.image_processor(image).to(self.device)
        image_embeds = self.get_image_features(**image_inputs)[0]
        if pq:
            desc = self.faiss_index.sa_encode(image_embeds.cpu().float().numpy())
            desc = torch.from_numpy(self.faiss_index.sa_decode(desc))
            image_embeds = desc.to(image_embeds.device, image_embeds.dtype)
            #inputs_embeds = product_quantization(inputs_embeds, segments, self.faiss_index, self.apply_asym)
        return BatchFeature(data={"image_embeds": image_embeds, **image_inputs})

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        pixel_values_videos (`torch.FloatTensor` of shape `(seq_length, num_channels * temporal_size * image_size * image_size)):
            The tensors corresponding to the input videos. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`Qwen2VLImageProcessor.__call__`] for details. [`Qwen2_5_VLProcessor`] uses
            [`Qwen2VLImageProcessor`] for processing videos.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        if image_embeds is not None:
            n_image_tokens = (input_ids == self.config.image_token_id).sum()
            n_image_features = image_embeds.shape[0]
            if (
                    not is_torchdynamo_compiling()
                    and n_image_tokens != n_image_features
            ):
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0)
            n_video_tokens = (input_ids == self.config.video_token_id).sum()
            n_video_features = video_embeds.shape[0]
            if (
                    not is_torchdynamo_compiling()
                    and n_video_tokens != n_video_features
            ):
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
                
                if not self.modes["none"]:
                    if self.save_tokens:
                        filename = os.path.join(
                            self.save_dir, str(self.sample_idx) + ".h5"
                        )
                        image_embeds_np = (
                            image_embeds.detach().cpu().to(torch.float32).numpy()
                        )
                        with h5py.File(filename, "w") as f:
                            f.create_dataset(
                                "tokens",
                                data=image_embeds_np,
                                compression="gzip",
                                compression_opts=5,
                            )
                        self.sample_idx += 1

                    start_indices = (input_ids[0] == START_IMAGE).nonzero(
                        as_tuple=True
                    )[0]
                    end_indices = (input_ids[0] == END_IMAGE).nonzero(as_tuple=True)[0]
                    # First image token indices
                    first_start = start_indices[0].item()
                    first_end = end_indices[0].item()

                    # Second image token indices
                    second_start = start_indices[1].item()
                    second_end = end_indices[1].item()

                    segments = [(first_start, first_end), (second_start, second_end)]

                    if self.modes["sampling"] or self.modes["pooling"]:
                        mode = "sampling" if self.modes["sampling"] else "pooling"
                        replacement = self.compression_config[mode][
                            "replacement_method"
                        ]
                        inputs_embeds, input_ids, attention_mask, position_ids = (
                            sampling_and_pooling_2D(
                                segments,
                                image_grid_thw,
                                inputs_embeds,
                                input_ids,
                                attention_mask,
                                position_ids,
                                mode,
                                replacement,
                                self.apply_asym,
                            )
                        )

                    elif self.modes["grouping"] or self.modes["clustering"]:
                        mode = "grouping" if self.modes["grouping"] else "clustering"
                        group_size = self.compression_config["grouping"]["group_size"]
                        n_clusters = self.compression_config["clustering"]["n_clusters"]
                        reduce_method = self.compression_config[mode]["reduce_method"]
                        inputs_embeds = grouping_and_clustering(
                            inputs_embeds,
                            segments,
                            mode,
                            group_size,
                            n_clusters,
                            reduce_method,
                            self.apply_asym,
                        )

                    else:
                        pass

            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
