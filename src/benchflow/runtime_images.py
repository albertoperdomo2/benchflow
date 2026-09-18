from __future__ import annotations


_INFERENCE_SIM_REPOSITORY = "llm-d-inference-sim"


def image_repository_basename(image: str) -> str:
    """Return an OCI image repository basename without its tag or digest."""
    reference = str(image or "").strip().split("@", 1)[0]
    basename = reference.rsplit("/", 1)[-1]
    if ":" in basename:
        basename = basename.rsplit(":", 1)[0]
    return basename.lower()


def is_inference_sim_image(image: str) -> bool:
    """Whether an image is the supported llm-d vLLM-compatible simulator."""
    return image_repository_basename(image) == _INFERENCE_SIM_REPOSITORY
