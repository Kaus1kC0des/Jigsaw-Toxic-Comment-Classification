from transformers import BertForSequenceClassification

class ToxicCommentModel(BertForSequenceClassification):
    def __init__(self, num_labels=6, **kwargs):
        super().__init__(config=self._get_config(num_labels, **kwargs))

    @staticmethod
    def _get_config(num_labels, **kwargs):
        from transformers import BertConfig
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=num_labels, **kwargs)
        return config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        return super().forward(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)