import os
import sys
import argparse
import time
import hashlib
import json
import yaml
from pathlib import Path
from types import SimpleNamespace
from UVR import *

# --- Assume these modules are in the correct relative paths ---
# Add the base path to sys.path to ensure imports work correctly
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_PATH)

from separate import SeperateMDX, SeperateMDXC, save_format, cuda_available, mps_available, clear_gpu_cache
from ml_collections import ConfigDict

# --- Constants ---
MDX_ARCH_TYPE = 'MDX_Net'
MODELS_DIR = os.path.join(BASE_PATH, 'models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
MDX_HASH_DIR = os.path.join(MDX_MODELS_DIR, 'model_data')
MDX_HASH_JSON = os.path.join(MDX_HASH_DIR, 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_HASH_DIR, 'mdx_c_configs')

# --- Helper Functions ---

def find_single_mdx_model():
    """
    Finds the single MDX model in the models directory.
    Exits if zero or more than one model is found.
    """
    print("Searching for MDX model...")
    models_found = []
    for ext in ('*.onnx', '*.ckpt'):
        models_found.extend(Path(MDX_MODELS_DIR).glob(ext))

    if len(models_found) == 0:
        print(f"Error: No MDX model (.onnx or .ckpt) found in '{MDX_MODELS_DIR}'.")
        sys.exit(1)
    
    if len(models_found) > 1:
        print(f"Error: More than one MDX model found in '{MDX_MODELS_DIR}'. Please ensure only one model is present.")
        for model_path in models_found:
            print(f" - {model_path.name}")
        sys.exit(1)
        
    model_path = models_found[0]
    print(f"Found model: {model_path.name}")
    return str(model_path)

def get_device():
    """
    Determines the processing device (GPU if available).
    """
    if cuda_available:
        print("CUDA GPU available. Using CUDA.")
        return 'cuda'
    if mps_available:
        print("Apple Silicon GPU available. Using MPS.")
        return 'mps'
    
    print("Error: No compatible GPU (CUDA or MPS) found. This CLI is configured to run on GPU only.")
    sys.exit(1)

def create_model_data(model_path, device):
    """
    Creates a model_data object with hardcoded defaults and model-specific parameters.
    """
    model_data = SimpleNamespace()
    model_data.model_basename = Path(model_path).stem
    model_data.primary_model_name = model_data.model_basename
    model_data.DEVERBER_MODEL = DEVERBER_MODEL_PATH
    model_data.DENOISER_MODEL = DENOISER_MODEL_PATH
    model_data.is_deverb_vocals = False
    model_data.deverb_vocal_opt = 'None'
    model_data.is_deverb_vocals = False
    model_data.is_denoise_model = False
    model_data.is_gpu_conversion = 0 if True else -1
    model_data.is_use_opencl = False
    model_data.is_primary_stem_only = False
    model_data.is_secondary_stem_only = False
    # model_data.is_denoise = True if not denoise_option == DENOISE_NONE else False
    model_data.is_mdx_c_seg_def = False
    model_data.mdx_batch_size = 1
    model_data.mdxnet_stem_select = 'Vocals'
    model_data.overlap = 0.25
    model_data.overlap_mdx = 0.25
    model_data.overlap_mdx23 = 0
    model_data.semitone_shift = 0
    model_data.is_pitch_change = False
    model_data.is_match_frequency_pitch = False
    model_data.is_mdx_ckpt = False
    model_data.is_mdx_c = False
    model_data.is_mdx_combine_stems = False
    model_data.mdx_c_configs = None
    model_data.mdx_model_stems = []
    model_data.mdx_dim_f_set = None
    model_data.mdx_dim_t_set = None
    model_data.mdx_stem_count = 1
    model_data.compensate = None
    model_data.mdx_n_fft_scale_set = None
    model_data.wav_type_set = 'PCM_16'
    model_data.device_set = device
    model_data.mp3_bit_set = '192k'
    model_data.save_format = 'wav'
    model_data.is_invert_spec = False
    model_data.is_mixer_mode = False
    model_data.demucs_stems = VOCAL_STEM
    model_data.is_demucs_combine_stems = False
    model_data.demucs_source_list = []
    model_data.demucs_stem_count = 0
    model_data.mixer_path = MDX_MIXER_PATH
    model_data.model_name = Path(model_path).stem
    model_data.process_method = MDX_ARCH_TYPE
    model_data.model_status = True
    model_data.primary_stem = None
    model_data.secondary_stem = None
    model_data.primary_stem_native = None
    model_data.is_ensemble_mode = False
    model_data.ensemble_primary_stem = None
    model_data.ensemble_secondary_stem = None
    model_data.primary_model_primary_stem = None
    model_data.is_secondary_model = False
    model_data.secondary_model = None
    model_data.secondary_model_scale = None
    model_data.demucs_4_stem_added_count = 0
    model_data.is_demucs_4_stem_secondaries = False
    model_data.is_4_stem_ensemble = False
    model_data.pre_proc_model = None
    model_data.pre_proc_model_activated = False
    model_data.is_pre_proc_model = False
    model_data.is_dry_check = False
    model_data.model_samplerate = 44100
    model_data.model_capacity = 32, 128
    model_data.is_vr_51_model = False
    model_data.is_demucs_pre_proc_model_inst_mix = False
    model_data.manual_download_Button = None
    model_data.secondary_model_4_stem = []
    model_data.secondary_model_4_stem_scale = []
    model_data.secondary_model_4_stem_names = []
    model_data.secondary_model_4_stem_model_names_list = []
    model_data.all_models = []
    model_data.secondary_model_other = None
    model_data.secondary_model_scale_other = None
    model_data.secondary_model_bass = None
    model_data.secondary_model_scale_bass = None
    model_data.secondary_model_drums = None
    model_data.secondary_model_scale_drums = None
    model_data.is_multi_stem_ensemble = False
    model_data.is_karaoke = False
    model_data.is_bv_model = False
    model_data.bv_model_rebalance = 0
    model_data.is_sec_bv_rebalance = False
    model_data.is_change_def = False
    model_data.model_hash_dir = None
    model_data.is_get_hash_dir_only = False
    model_data.is_secondary_model_activated = False
    model_data.vocal_split_model = None
    model_data.is_vocal_split_model = False
    model_data.is_vocal_split_model_activated = False
    model_data.is_save_inst_vocal_splitter = False
    model_data.is_inst_only_voc_splitter = False
    model_data.is_save_vocal_only = False

    # --- Basic Info ---
    model_data.model_path = model_path
    model_data.process_method = MDX_ARCH_TYPE
    model_data.model_name = Path(model_path).stem
    model_data.device_set = device
    model_data.is_gpu_conversion = 0  # This is an internal flag used by the separator

    # --- Hardcoded Defaults ---
    model_data.mdx_segment_size = 256
    model_data.overlap_mdx = 0.25
    model_data.overlap_mdx23 = 8
    model_data.margin = 44100
    model_data.mdx_batch_size = 1
    model_data.is_invert_spec = False
    model_data.is_denoise = False
    model_data.is_normalization = True
    model_data.wav_type_set = 'PCM_16'
    model_data.save_format = 'WAV'
    model_data.mp3_bit_set = '320k'
    model_data.is_primary_stem_only = False
    model_data.is_secondary_stem_only = False
    model_data.mdxnet_stem_select = 'Vocals'
    model_data.is_mdx23_combine_stems = False

    # --- Model-Specific Parameters ---
    model_data.is_mdx_c = False
    model_data.mdx_c_configs = None
    model_data.mdx_model_stems = []

    print("Reading model parameters...")
    try:
        with open(model_path, 'rb') as f:
            f.seek(-10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except (IOError, ValueError):
        with open(model_path, 'rb') as f:
            model_hash = hashlib.md5(f.read()).hexdigest()

    if not os.path.exists(MDX_HASH_JSON):
        print(f"Error: Model data file not found at '{MDX_HASH_JSON}'.")
        sys.exit(1)

    with open(MDX_HASH_JSON, 'r') as d:
        mdx_hash_mapper = json.load(d)

    params = mdx_hash_mapper.get(model_hash)
    if not params:
        print(f"Error: Model '{model_data.model_name}' with hash '{model_hash}' is not registered in '{MDX_HASH_JSON}'.")
        print("Please run the GUI version once to register the model or add its parameters manually.")
        sys.exit(1)

    if "config_yaml" in params:
        model_data.is_mdx_c = True
        config_path = os.path.join(MDX_C_CONFIG_PATH, params["config_yaml"])
        if not os.path.isfile(config_path):
            print(f"Error: MDX-C config file '{params['config_yaml']}' not found.")
            sys.exit(1)
        with open(config_path) as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        model_data.mdx_c_configs = config
        
        if config.training.target_instrument:
            model_data.primary_stem = config.training.target_instrument
            model_data.mdx_model_stems = [config.training.target_instrument]
        else:
            model_data.mdx_model_stems = config.training.instruments
            model_data.primary_stem = model_data.mdxnet_stem_select
        
        model_data.mdx_stem_count = len(model_data.mdx_model_stems)
    else:
        model_data.compensate = params.get("compensate", 1.035)
        model_data.mdx_dim_f_set = params["mdx_dim_f_set"]
        model_data.mdx_dim_t_set = params["mdx_dim_t_set"]
        model_data.mdx_n_fft_scale_set = params["mdx_n_fft_scale_set"]
        model_data.primary_stem = params["primary_stem"]
        model_data.mdx_stem_count = 1

    model_data.secondary_stem = 'Instrumental' if model_data.primary_stem == 'Vocals' else 'No ' + model_data.primary_stem
    print(f"Model configured for primary stem: '{model_data.primary_stem}'")
    
    return model_data

def main():
    parser = argparse.ArgumentParser(description="Command-line interface for MDX-Net audio separation.")
    parser.add_argument('input_file', type=str, help="Path to the input audio file.")
    parser.add_argument('output_dir', type=str, help="Path to the directory where output files will be saved.")
    args = parser.parse_args()

    # --- Validate Inputs ---
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Setup ---
    stime = time.perf_counter()
    model_path = find_single_mdx_model()
    device = get_device()
    model_data = create_model_data(model_path, device)

    # --- Prepare for Separation ---
    audio_file_base = Path(args.input_file).stem
    
    def write_to_console(progress_text):
        print(f"\r{progress_text}", end='', flush=True)

    def set_progress_bar(step, inference_iterations=0):
        # A simple text-based progress bar
        progress = int(step * 50) # 50 characters width
        bar = '[' + '#' * progress + '-' * (50 - progress) + ']'
        write_to_console(f"Processing... {bar} {int(step*100)}%")

    process_data = {
        'model_data': model_data,
        'export_path': args.output_dir,
        'audio_file_base': audio_file_base,
        'audio_file': args.input_file,
        'set_progress_bar': set_progress_bar,
        'write_to_console': write_to_console,
        # These are not used by MDX but are part of the original signature
        'process_iteration': lambda: None,
        'cached_source_callback': lambda *args: (None, None),
        'cached_model_source_holder': lambda *args: None,
        'list_all_models': [],
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False
    }

    # --- Run Separation ---
    print(f"\nStarting separation for '{Path(args.input_file).name}'...")
    try:
        if model_data.is_mdx_c:
            separator = SeperateMDXC(model_data, process_data)
        else:
            separator = SeperateMDX(model_data, process_data)
        
        separator.seperate()
        
        print("\nSeparation complete.")
    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_gpu_cache()
        
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(int(time.perf_counter() - stime)))
    print(f"Total time elapsed: {time_elapsed}")

if __name__ == "__main__":
    main()
