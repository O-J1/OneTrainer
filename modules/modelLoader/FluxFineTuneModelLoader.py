from modules.model.FluxModel import FluxModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelLoader.flux.FluxEmbeddingLoader import FluxEmbeddingLoader
from modules.modelLoader.flux.FluxModelLoader import FluxModelLoader
from modules.modelLoader.mixin.InternalModelLoaderMixin import InternalModelLoaderMixin
from modules.modelLoader.mixin.ModelSpecModelLoaderMixin import ModelSpecModelLoaderMixin
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class FluxFineTuneModelLoader(
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
            case ModelType.FLUX_DEV_1:
                return "resources/sd_model_spec/flux_dev_1.0.json"
            case ModelType.FLUX_FILL_DEV_1:
                return "resources/sd_model_spec/flux_dev_fill_1.0.json"
            case _:
                return None

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> FluxModel | None:
        base_model_loader = FluxModelLoader()
        embedding_loader = FluxEmbeddingLoader()

        model = FluxModel(model_type=model_type)

        self._load_internal_data(model, model_names.base_model)
        model.model_spec = self._load_default_model_spec(model_type)

        base_model_loader.load(model, model_type, model_names, weight_dtypes)
        embedding_loader.load(model, model_names.base_model, model_names)

        return model
