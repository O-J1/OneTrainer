from modules.model.HunyuanVideoModel import HunyuanVideoModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoEmbeddingLoader import HunyuanVideoEmbeddingLoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoLoRALoader import HunyuanVideoLoRALoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoModelLoader import HunyuanVideoModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class HunyuanVideoLoRAModelLoader(
    BaseModelLoader,
    ModelSpecModelLoaderMixin,
    InternalModelLoaderMixin,
):
    def __init__(self):
        super().__init__()

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.HUNYUAN_VIDEO:
                return "resources/sd_model_spec/hunyuan_video-lora.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> HunyuanVideoModel | None:
        base_model_loader = HunyuanVideoModelLoader()
        lora_model_loader = HunyuanVideoLoRALoader()
        embedding_loader = HunyuanVideoEmbeddingLoader()

        model = HunyuanVideoModel(model_type=model_type)
        self._load_internal_data(model, model_names.lora)
        model.model_spec = self._load_default_model_spec(model_type)

        if model_names.base_model is not None:
            base_model_loader.load(model, model_type, model_names, weight_dtypes)
        lora_model_loader.load(model, model_names)
        embedding_loader.load(model, model_names.lora, model_names)

        return model
