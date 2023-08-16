import torch.cuda
import langcodes
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

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

        self.supported_source_lang = "en"
        self.supported_target_langs = {"es": {"es": None}}

        self.logger.info(f"Cuda is avaliable: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"Number of avaliable GPUs: {torch.cuda.device_count()}")

        with Timer(f"LaLlama model initialization", self.logger):
            self.tokenizer = AutoTokenizer.from_pretrained(lallama_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                lallama_model, load_in_8bit=True
            )

        self.logger.info(f"LaLlama translator model is using {self.model.device}")

    def _translate_lines(self, lines):
        translated_lines = []
        for line in lines:
            # Add target language prefix for multilingual Marian
            line_with_prefix = f"### System:\nTraduce este texto al espa_ol\n\n### User:\n{line}\n\n### Response:\n"

            self.logger.info(line_with_prefix)

            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = "left"

            encoded = self.tokenizer(
                line_with_prefix, return_tensors="pt", padding=False, truncation=False
            )

            if (
                encoded["input_ids"].shape[1]
                >= self.model.config.max_position_embeddings
            ):
                raise Exception(
                    f"Input length ({encoded['input_ids'].shape[1]} tokens) is greater "
                    f"than model maximum length ({self.model.config.max_length} tokens), "
                    f"consider using sentence separation before translation."
                )

            generation_config = GenerationConfig(
                **{
                    "temperature": 0.95,
                    "top_p": 0.9,
                    "top_k": 50,
                    "num_beams": 1,
                    "use_cache": True,
                    "repetition_penalty": 1.2,
                    "max_new_tokens": 4096,
                    "do_sample": True,
                    "pad_token_id": 32000,
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                }
            )

            encoded.to(self.model.device)
            generated_tokens = self.model.generate(
                **encoded, generation_config=generation_config
            )
            self.logger.info(generated_tokens)
            # remove the input tokens
            output_tokens = generated_tokens[:, len(encoded[0]) :]

            # Since we're processing one line at a time, we use decode instead of batch_decode
            translated_line = self.tokenizer.decode(
                output_tokens[0], skip_special_tokens=True
            )

            translated_lines.append(translated_line)

        return translated_lines

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
