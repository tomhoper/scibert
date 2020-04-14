import scibert.dataset_readers.classification_dataset_reader
import scibert.models.text_classifier

from allennlp.predictors import Predictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
from allennlp.common.util import JsonDict, sanitize
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.fields import MetadataField


@Predictor.register('bert_crf_tagger')
class SciPredictor(SentenceTaggerPredictor):
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "...","docid": int, "sentid": int}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        docid = json_dict["docid"]
        sentid = json_dict["sentid"]

        tokens = self._tokenizer.split_words(sentence)
        instance = self._dataset_reader.text_to_instance(tokens)
        docid_field  = MetadataField(docid)
        sentid_field  = MetadataField(sentid)

        instance.add_field(field_name="docid",field=docid_field)
        instance.add_field(field_name="sentid",field=sentid_field)

        return instance

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        del outputs["logits"]
        del outputs["mask"]
        outputs["docid"] = instance.fields["docid"].metadata
        outputs["sentid"] = instance.fields["sentid"].metadata
        return sanitize(outputs)
    
    
    @overrides
    def predict_batch_instance(self, instances: Instance) -> JsonDict:
        outputs = self._model.forward_on_instances(instances)
        for i in range(len(outputs)):
            del outputs[i]["logits"]
            del outputs[i]["mask"]
            outputs[i] = sanitize(outputs[i])
            outputs[i]["docid"] = instances[i].fields["docid"].metadata
            outputs[i]["sentid"] = instances[i].fields["sentid"].metadata

        return outputs



    