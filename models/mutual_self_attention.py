# *************************************************************************
# Copyright (2024) Bytedance Inc.
#
# Copyright (2024) LightningDrag Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

import torch
import torch.nn.functional as F

from typing import Any, Dict, Optional

from diffusers.models.attention import BasicTransformerBlock

from drag_util import torch_dfs


class ReferenceAttentionControl():
    
    def __init__(self, 
                 unet,
                 mode="write",
                 do_classifier_free_guidance=False,
                 attention_auto_machine_weight = float('inf'),
                 gn_auto_machine_weight=1.0,
                 style_fidelity=1.0,
                 reference_attn=True,
                 reference_adain=False,
                 fusion_blocks="midup",
                 batch_size=1,
                 num_images_per_prompt=1, 
                 skip_cfg_appearance_encoder=False,
                 ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full", "up"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode, 
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            fusion_blocks=fusion_blocks,
            skip_cfg_appearance_encoder=skip_cfg_appearance_encoder, 
        )

    def register_reference_hooks(
            self, 
            mode, 
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            dtype=torch.float16,
            batch_size=1, 
            num_images_per_prompt=1, 
            device=torch.device("cpu"), 
            fusion_blocks='midup',
            skip_cfg_appearance_encoder=False,
        ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        skip_cfg_appearance_encoder = skip_cfg_appearance_encoder
        num_images_per_prompt = num_images_per_prompt
        dtype=dtype
        if do_classifier_free_guidance:
            uc_mask = torch.Tensor(
                [1 if not skip_cfg_appearance_encoder else 0] * batch_size * num_images_per_prompt \
                + [0] * batch_size * num_images_per_prompt
            ).to(device).bool()
        else:
            uc_mask = torch.Tensor(
                [0] * batch_size * num_images_per_prompt * 2
            ).to(device).bool()
            

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
        ):
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    hidden_states_c = self.attn1(
                        norm_hidden_states, 
                        encoder_hidden_states=torch.cat([norm_hidden_states] + self.bank, dim=1),
                        attention_mask=attention_mask,
                        handle_embeddings=self.handle_embeddings,
                        target_embeddings=self.target_embeddings,
                        uc_mask=uc_mask, # important, identify which key/value should add embeddings
                        ) + hidden_states

                    # if do_classifier_free_guidance and not skip_cfg_appearance_encoder:
                    #     _uc_mask = uc_mask.clone()
                    #     hidden_states_c[_uc_mask] = self.attn1(
                    #         norm_hidden_states[_uc_mask],
                    #         encoder_hidden_states=norm_hidden_states[_uc_mask],
                    #         attention_mask=attention_mask,
                    #         handle_embeddings=None,
                    #         target_embeddings=None,
                    #     ) + hidden_states[_uc_mask]
                    hidden_states = hidden_states_c

                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                            )
                            + hidden_states
                        )

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    return hidden_states
                
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.fusion_blocks == "midup":
            attn_modules = [module for module in
                            (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks))
                            if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "up":
            attn_modules = [module for module in torch_dfs(self.unet.up_blocks)
                            if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            attn_modules = [module for module in torch_dfs(self.unet)
                            if isinstance(module, BasicTransformerBlock)]            
        attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])

        for i, module in enumerate(attn_modules):
            module._original_inner_forward = module.forward
            module.forward = hacked_basic_transformer_inner_forward.__get__(module, BasicTransformerBlock)
            module.bank = []
            module.handle_embeddings = None
            module.target_embeddings = None

    def update(self, writer, dtype=torch.float16):
        if self.fusion_blocks == "midup":
            reader_attn_modules = [module for module in
                                   (torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks))
                                   if isinstance(module, BasicTransformerBlock)]
            writer_attn_modules = [module for module in
                                   (torch_dfs(writer.unet.mid_block) + torch_dfs(writer.unet.up_blocks))
                                   if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "up":
            reader_attn_modules = [module for module in torch_dfs(self.unet.up_blocks)
                                   if isinstance(module, BasicTransformerBlock)]
            writer_attn_modules = [module for module in torch_dfs(writer.unet.up_blocks)
                                   if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            reader_attn_modules = [module for module in torch_dfs(self.unet) 
                                   if isinstance(module, BasicTransformerBlock)]
            writer_attn_modules = [module for module in torch_dfs(writer.unet)
                                   if isinstance(module, BasicTransformerBlock)]
        reader_attn_modules = sorted(reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])    
        writer_attn_modules = sorted(writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        for r, w in zip(reader_attn_modules, writer_attn_modules):
            r.bank = [v.clone().to(dtype) for v in w.bank]

    def clear(self):
        if self.fusion_blocks == "midup":
            attn_modules = [module for module in
                            (torch_dfs(self.unet.mid_block)+torch_dfs(self.unet.up_blocks))
                            if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "up":
            attn_modules = [module for module in
                            torch_dfs(self.unet.up_blocks)
                            if isinstance(module, BasicTransformerBlock)]
        elif self.fusion_blocks == "full":
            attn_modules = [module for module in torch_dfs(self.unet) if isinstance(module, BasicTransformerBlock)]
        attn_modules = sorted(attn_modules, key=lambda x: -x.norm1.normalized_shape[0])
        for m in attn_modules:
            m.bank.clear()
