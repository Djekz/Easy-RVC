import os
import json

import librosa
import soundfile
import numpy as np

import gradio as gr
from UVR_interface import (
    root,
    UVRInterface,
    VR_MODELS_DIR,
    MDX_MODELS_DIR,
    DEMUCS_MODELS_DIR,
)
from gui_data.constants import *
from typing import List, Dict, Callable, Union


class UVRWebUI:
    def __init__(self, uvr: UVRInterface, online_data_path: str) -> None:
        self.uvr = uvr
        self.models_url = self.get_models_url(online_data_path)
        self.define_layout()

        self.input_temp_dir = "__temp"
        self.export_path = "out"
        if not os.path.exists(self.input_temp_dir):
            os.mkdir(self.input_temp_dir)

    def get_models_url(self, models_info_path: str) -> Dict[str, Dict]:
        with open(models_info_path, "r") as f:
            online_data = json.loads(f.read())
        models_url = {}
        for arch, download_list_key in zip(
            [VR_ARCH_TYPE, MDX_ARCH_TYPE], ["vr_download_list", "mdx_download_list"]
        ):
            models_url[arch] = {
                model: NORMAL_REPO + model_path
                for model, model_path in online_data[download_list_key].items()
            }
        models_url[DEMUCS_ARCH_TYPE] = online_data["demucs_download_list"]
        return models_url

    def get_local_models(self, arch: str) -> List[str]:
        model_config = {
            VR_ARCH_TYPE: (VR_MODELS_DIR, ".pth"),
            MDX_ARCH_TYPE: (MDX_MODELS_DIR, ".onnx"),
            DEMUCS_ARCH_TYPE: (DEMUCS_MODELS_DIR, ".yaml"),
        }
        try:
            model_dir, suffix = model_config[arch]
        except KeyError:
            raise ValueError(f"Unkown arch type: {arch}")
        return [
            os.path.splitext(f)[0] for f in os.listdir(model_dir) if f.endswith(suffix)
        ]

    def set_arch_setting_value(self, arch: str, setting1, setting2):
        if arch == VR_ARCH_TYPE:
            root.window_size_var.set(setting1)
            root.aggression_setting_var.set(setting2)
        elif arch == MDX_ARCH_TYPE:
            root.mdx_batch_size_var.set(setting1)
            root.compensate_var.set(setting2)
        elif arch == DEMUCS_ARCH_TYPE:
            pass

    def arch_select_update(self, arch: str) -> List[Dict]:
        choices = self.get_local_models(arch)
        if arch == VR_ARCH_TYPE:
            model_update = self.model_choice.update(
                choices=choices, value=CHOOSE_MODEL, label=SELECT_VR_MODEL_MAIN_LABEL
            )
            setting1_update = self.arch_setting1.update(
                choices=VR_WINDOW,
                label=WINDOW_SIZE_MAIN_LABEL,
                value=root.window_size_var.get(),
            )
            setting2_update = self.arch_setting2.update(
                choices=VR_AGGRESSION,
                label=AGGRESSION_SETTING_MAIN_LABEL,
                value=root.aggression_setting_var.get(),
            )
        elif arch == MDX_ARCH_TYPE:
            model_update = self.model_choice.update(
                choices=choices, value=CHOOSE_MODEL, label=CHOOSE_MDX_MODEL_MAIN_LABEL
            )
            setting1_update = self.arch_setting1.update(
                choices=BATCH_SIZE,
                label=BATCHES_MDX_MAIN_LABEL,
                value=root.mdx_batch_size_var.get(),
            )
            setting2_update = self.arch_setting2.update(
                choices=VOL_COMPENSATION,
                label=VOL_COMP_MDX_MAIN_LABEL,
                value=root.compensate_var.get(),
            )
        elif arch == DEMUCS_ARCH_TYPE:
            model_update = self.model_choice.update(
                choices=choices,
                value=CHOOSE_MODEL,
                label=CHOOSE_DEMUCS_MODEL_MAIN_LABEL,
            )
            raise gr.Error(f"{DEMUCS_ARCH_TYPE} not implempted")
        else:
            raise gr.Error(f"Unkown arch type: {arch}")
        return [model_update, setting1_update, setting2_update]

    def model_select_update(
        self, arch: str, model_name: str
    ) -> List[Union[str, Dict, None]]:
        if model_name == CHOOSE_MODEL:
            return [None for _ in range(4)]
        (model,) = self.uvr.assemble_model_data(model_name, arch)
        if not model.model_status:
            raise gr.Error(f"Cannot get model data, model hash = {model.model_hash}")

        stem1_check_update = self.primary_stem_only.update(
            label=f"{model.primary_stem} Only"
        )
        stem2_check_update = self.secondary_stem_only.update(
            label=f"{model.secondary_stem} Only"
        )
        stem1_out_update = self.primary_stem_out.update(
            label=f"Output {model.primary_stem}"
        )
        stem2_out_update = self.secondary_stem_out.update(
            label=f"Output {model.secondary_stem}"
        )

        return [
            stem1_check_update,
            stem2_check_update,
            stem1_out_update,
            stem2_out_update,
        ]

    def checkbox_set_root_value(self, checkbox: gr.Checkbox, root_attr: str):
        checkbox.change(
            lambda value: root.__getattribute__(root_attr).set(value), inputs=checkbox
        )

    def set_checkboxes_exclusive(
        self,
        checkboxes: List[gr.Checkbox],
        pure_callbacks: List[Callable],
        exclusive_value=True,
    ):
        def exclusive_onchange(i, callback_i):
            def new_onchange(*check_values):
                if check_values[i] == exclusive_value:
                    return_values = []
                    for j, value_j in enumerate(check_values):
                        if j != i and value_j == exclusive_value:
                            return_values.append(not exclusive_value)
                        else:
                            return_values.append(value_j)
                else:
                    return_values = check_values
                callback_i(check_values[i])
                return return_values

            return new_onchange

        for i, (checkbox, callback) in enumerate(zip(checkboxes, pure_callbacks)):
            checkbox.change(
                exclusive_onchange(i, callback), inputs=checkboxes, outputs=checkboxes
            )

    def process(
        self,
        input_audio,
        input_filename,
        model_name,
        arch,
        setting1,
        setting2,
        progress=gr.Progress(),
    ):
        def set_progress_func(step, inference_iterations=0):
            progress_curr = step + inference_iterations
            progress(progress_curr)

        sampling_rate, audio = input_audio
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        input_path = os.path.join(self.input_temp_dir, input_filename)
        soundfile.write(input_path, audio, sampling_rate, format="wav")

        self.set_arch_setting_value(arch, setting1, setting2)

        seperator = uvr.process(
            model_name=model_name,
            arch_type=arch,
            audio_file=input_path,
            export_path=self.export_path,
            is_model_sample_mode=root.model_sample_mode_var.get(),
            set_progress_func=set_progress_func,
        )

        primary_audio = None
        secondary_audio = None
        msg = ""
        if not seperator.is_secondary_stem_only:
            primary_stem_path = os.path.join(
                seperator.export_path,
                f"{seperator.audio_file_base}_({seperator.primary_stem}).wav",
            )
            audio, rate = soundfile.read(primary_stem_path)
            primary_audio = (rate, audio)
            msg += f"{seperator.primary_stem} saved at {primary_stem_path}\n"
        if not seperator.is_primary_stem_only:
            secondary_stem_path = os.path.join(
                seperator.export_path,
                f"{seperator.audio_file_base}_({seperator.secondary_stem}).wav",
            )
            audio, rate = soundfile.read(secondary_stem_path)
            secondary_audio = (rate, audio)
            msg += f"{seperator.secondary_stem} saved at {secondary_stem_path}\n"

        os.remove(input_path)

        return primary_audio, secondary_audio, msg
