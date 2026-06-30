DEFAULT_FPS = 1
DEFAULT_CHUNKS = 4
DEFAULT_PARALLEL_CHUNKS = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.55
DEFAULT_MERGE_GAP_SEC = 1.5
DEFAULT_MIN_INTERVAL_SEC = 1.0
DEFAULT_DET_SIZE = 640
DEFAULT_MODEL_NAME = "buffalo_l"
DEFAULT_CTX_ID = 0
DEFAULT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
DEFAULT_SAVE_DEBUG = False
DEFAULT_DEBUG_DIR = "./debug"
DEFAULT_OVERLAP_SEC = 1.0

# Second-pass vision-LLM verification.
DEFAULT_VERIFY_PROMPT = (
    "You are verifying whether two images show the SAME person. "
    "The FIRST image is a reference face. The SECOND image is a face cropped "
    "from a video frame. Answer with exactly YES if they are the same "
    "individual, or NO if they are different people or you are not sure. "
    "Begin your answer with YES or NO."
)
DEFAULT_VERIFY_MAX_TOKENS = 64
