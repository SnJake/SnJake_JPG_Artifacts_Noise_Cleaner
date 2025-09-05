from .jpg_noise_remover import SnJakeArtifactsRemover

NODE_CLASS_MAPPINGS = {
    "SnJakeArtifactsRemover": SnJakeArtifactsRemover,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SnJakeArtifactsRemover": "😎 JPG & Noise Remover",
}


WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]


print("### SnJake Nodes Initialized ###")










