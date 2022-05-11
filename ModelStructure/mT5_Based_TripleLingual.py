import torch
from torch.nn import CrossEntropyLoss
from transformers import MT5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, \
    Seq2SeqLMOutput, add_start_docstrings_to_model_forward, replace_return_docstrings, warnings, \
    __HEAD_MASK_WARNING_MSG, BaseModelOutput, T5LayerCrossAttention


class MT5_TripleLingual_LateFusion_FinalEmbedding(MT5ForConditionalGeneration):
    def __init__(self, config):
        super(MT5_TripleLingual_LateFusion_FinalEmbedding, self).__init__(config)
        self.lm_head_neo = torch.nn.Linear(config.d_model * 2, config.vocab_size, bias=False)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids_x=None,
            #########################################
            input_ids_y=None,
            #########################################
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        # Convert encoder inputs in embeddings if needed
        encoder_outputs_x = self.encoder(
            input_ids=input_ids_x, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        encoder_outputs_y = self.encoder(
            input_ids=input_ids_y, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )

        ##########################################

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        hidden_states = encoder_outputs_x[0]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        sequence_output_x = decoder_outputs[0]

        hidden_states = encoder_outputs_y[0]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        sequence_output_y = decoder_outputs[0]

        sequence_output = torch.cat([sequence_output_x, sequence_output_y], dim=-1)
        lm_logits = self.lm_head_neo(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )


class MT5_TripleLingual_LateFusion_Project(MT5ForConditionalGeneration):
    def __init__(self, config):
        super(MT5_TripleLingual_LateFusion_Project, self).__init__(config)
        self.lm_project_layer = torch.nn.Linear(config.d_model * 2, config.d_model, bias=False)
        print('Adding Project Layer')
        print(self.lm_project_layer)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids_x=None,
            #########################################
            input_ids_y=None,
            #########################################
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        # Convert encoder inputs in embeddings if needed
        encoder_outputs_x = self.encoder(
            input_ids=input_ids_x, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        encoder_outputs_y = self.encoder(
            input_ids=input_ids_y, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )

        ##########################################

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        hidden_states = encoder_outputs_x[0]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        sequence_output_x = decoder_outputs[0]

        hidden_states = encoder_outputs_y[0]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        sequence_output_y = decoder_outputs[0]

        sequence_output = torch.cat([sequence_output_x, sequence_output_y], dim=-1)

        project_result = self.lm_project_layer(sequence_output)
        lm_logits = self.lm_head(project_result)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )


class MT5_TripleLingual_HiddenLayerProject(MT5ForConditionalGeneration):
    def __init__(self, config):
        super(MT5_TripleLingual_HiddenLayerProject, self).__init__(config)
        # self.lm_project_layer = torch.nn.Linear(config.d_model * 2, config.d_model, bias=False)
        self.cross_lingual_attention = T5LayerCrossAttention(config)
        print('Adding Cross Hidden State Project Layer')
        print(self.cross_lingual_attention)

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids_x=None,
            #########################################
            input_ids_y=None,
            #########################################
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        # Convert encoder inputs in embeddings if needed
        encoder_outputs_x = self.encoder(
            input_ids=input_ids_x, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )
        encoder_outputs_y = self.encoder(
            input_ids=input_ids_y, attention_mask=attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, )

        ##########################################

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        # hidden_states = encoder_outputs_x[0]
        hidden_states = self.cross_lingual_attention(encoder_outputs_x[0], encoder_outputs_y[0])[0]
        ############################################################

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
