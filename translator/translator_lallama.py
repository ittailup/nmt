import torch.cuda
import langcodes
from transformers import AutoTokenizer, AutoModelForCausalLM

from translator.translator_base import TranslatorBase
from utils.utils import Timer, find_closest_language


class TranslatorLallama(TranslatorBase):
    @staticmethod
    def get_pretrained_model_name(source_language, target_language):
        known_models = {
            langcodes.Language.get("en"): {
                langcodes.Language.get("es"): "ittailup/lallama-13b-merged2",
            }
        }

        source_language = langcodes.Language.get(source_language)
        target_language = langcodes.Language.get(target_language)

        best_source_lang = find_closest_language(source_language, known_models.keys())
        if not best_source_lang:
            # We don't know pre-trained model for this source language
            return None

        models_for_source = known_models[best_source_lang]
        best_target_lang = find_closest_language(
            target_language, models_for_source.keys()
        )
        if not best_target_lang:
            # We don't know pre-trained model for this target language
            return None

        return models_for_source[best_target_lang]

    def __init__(
        self,
        lallama_model="ittailup/lallama-13b-merged2",
        device=None,
        max_batch_lines=8,
        max_batch_chars=2000,
        max_file_lines=60,
        verbose=False,
    ):

        super(TranslatorLallama, self).__init__(
            max_batch_lines=max_batch_lines,
            max_batch_chars=max_batch_chars,
            max_file_lines=max_file_lines,
            verbose=verbose,
        )

        self.logger.info(f"Cuda is avaliable: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"Number of avaliable GPUs: {torch.cuda.device_count()}")

        with Timer(f"LaLlama model initialization", self.logger):
            self.tokenizer = AutoTokenizer.from_pretrained(lallama_model)
            self.model = AutoModelForCausalLM.from_pretrained(lallama_model)

        self.supported_source_lang = langcodes.Language.get(self.tokenizer.source_lang)
        self.supported_target_langs = {
            langcodes.Language.get(lang): lang
            for lang in self.tokenizer.target_lang.split("+")
        }

        self.multilingual = len(self.supported_target_langs) > 1

        self.model.eval()
        if device:
            self.model.to(device)
        elif torch.cuda.is_available():
            self.model.to("cuda")

        self.logger.info(f"LaLlama translator model is using {self.model.device}")
        self.logger.info(f"Supported source language: {self.supported_source_lang}")
        self.logger.info(
            f"Supported target languages: {[str(lang) for lang in self.supported_target_langs.keys()]}"
        )

    def _translate_lines(self, lines):
        if self.multilingual:
            # For multilingual Marian we need to add target language to each line
            # https://huggingface.co/transformers/v2.9.1/model_doc/marian.html
            lines = [
                f"### System:\nTraduce este texto al español\n\n### User:\n{line}\n\n### Response:\n"
                for line in lines
            ]

        encoded = self.tokenizer(
            lines, return_tensors="pt", padding=True, truncation=False
        )

        if encoded["input_ids"].shape[1] >= self.model.config.max_length:
            raise Exception(
                f"Input length ({encoded['input_ids'].shape[1]} tokens) is greater "
                f"than model maximum length ({self.model.config.max_length} tokens), "
                f"consider using sentence separation before translation."
            )

        encoded.to(self.model.device)
        generated_tokens = self.model.generate(**encoded, num_beams=self.beam_size)
        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return result

    def _set_language_pair(self, source_language, target_language):
        # We don't need to specifically set source language for Marian.
        # Let's just check that the language is supported
        if not find_closest_language(source_language, [self.supported_source_lang]):
            raise Exception(
                f"Source language {source_language} is not supported. "
                f"The only supported source language is {self.supported_source_lang}."
            )

        # Let's try to find appropriate target language
        best_target_lang = find_closest_language(
            target_language, self.supported_target_langs.keys()
        )
        # Checking that best supported target language is not far away from the language that we asked to do
        if not best_target_lang:
            raise Exception(
                f"Target language {target_language} is not supported. "
                f"Supported target languages are {[str(lang) for lang in self.supported_target_langs.keys()]}."
            )
        # Storing target language tag to use during translation
        self.target_lang_tag = f">>{self.supported_target_langs[best_target_lang]}<<"

        return
