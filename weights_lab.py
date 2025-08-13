import ast
import dash
from dash import dcc
from dash import html
from enum import Enum
import re
from dash import dcc, html
from dash import dash_table
from dash.dependencies import Input
from dash.dependencies import Output
from dash.dependencies import State

import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from typing import Iterable, Tuple, Dict, List, Any

import argparse
import asyncio
import base64
import grpc
import os
import time
import threading
import sys
import dash_bootstrap_components as dbc
import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, MATCH, ALL, no_update
from dash.dependencies import Input, Output, State
import logging
import collections
import numpy as np

from collections import defaultdict

from dash.dash_table.Format import Format, Scheme
from scope_timer import ScopeTimer
from dataclasses import dataclass
from math import isqrt


logging.getLogger('werkzeug').setLevel(logging.ERROR)

_HYPERPARAM_COLUMNS = ["label", "type", "name", "value"]

_NEURONS_DF_COLUMNS = [
    "layer_id", "neuron_id", "Age", "RTrn", "REval", "ADiff",
    "RDiff", "Frozen", "Status", "layer_type", "highlighted"]

_LAYER_DF_COLUMNS = [
    "layer_id", "layer_type", "layer_name", "outgoing", "incoming",
    "kernel_size", "sorted_by"]

_METRICS_DF_COLUMNS = [
    "experiment_name", "model_age", "metric_name", "metric_value", "run_id"]

_ANNOTATIONS_DF_COLUMNS = [
    "experiment_name", "model_age", "annotation", "metadata"]

_SAMPLES_DF_COLUMNS = [
    "SampleId", "Target", "Prediction", "LastLoss", "Encounters", "Discarded" 
]

_PLOTS_COLOR_WHEEL = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Yellow-Green
    "#17becf",  # Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
    "#c49c94",  # Light Brown
    "#f7b6d2",  # Light Pink
    "#c7c7c7",  # Light Gray
    "#dbdb8d",  # Light Yellow-Green
    "#9edae5"   # Light Cyan
]

_DISPLAY_COLUMNS = [
    "SampleId", "Target", "Prediction", "LastLoss"
]

_BUTTON_STYLE = {
    'width': '6vw',
    'height': '8vh',
}

_DEFAULT_CHECKLIST_VALUES = [
    "neuron_id", "neuron_age", "trigger_rate_train", "Status"]


_LAYER_BUTTON_WIDTH = '3vw'
_LAYER_BUTTON_HEIGHT = '5vw'
_LAYER_BUTTON_FNT_SZ = '30px'
LYR_BASE_WIDTH = 140
WIDTH_PER_COLUMN = 80


# Custom CSS styles (keep the existing styles)
custom_styles = {
    'container': {
        'background': 'linear-gradient(135deg, #E0E5EC 0%, #B8C6DB 100%)',
        'min-height': '100vh',
        'padding': '20px',
        'font-family': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    'main_card': {
        'background': 'rgba(255, 255, 255, 0.95)',
        'backdrop-filter': 'blur(20px)',
        'border-radius': '20px',
        'box-shadow': '0 10px 20px rgba(0, 0, 0, 0.1)',
        'border': '1px solid rgba(255, 255, 255, 0.2)',
        'width': '95vw',      # Changed to 95% of viewport width
        'margin': '0 auto',
        'padding': '20px'
    },
    'header': {
        'display': 'flex',
        'justify-content': 'space-between',
        'align-items': 'center',
        'margin-bottom': '20px',
        'padding': '0 10px'
    },
    'logo': {
        'font-size': '24px',
        'font-weight': '700',
        'color': '#3B5998',
        'letter-spacing': '-0.5px',
        'margin': '0'
    },
    'config_section': {
        'padding': '0px'
    },
    'config_card': {
        'background': '#F7FAFC',
        'border-radius': '8px',
        'padding': '6px',
        'margin-bottom': '6px',
        'border': '1px solid #E2E8F0',
        'transition': 'all 0.3s ease'
    },
    'config_label': {
        'font-weight': '600',
        'color': '#4A5568',
        'font-size': '10px',
        'text-transform': 'uppercase',
        'letter-spacing': '0.5px',
        'width': '40%',
        'display': 'inline-block',
        'vertical-align': 'middle'
    },
    'input_style': {
        'border-radius': '4px',
        'border': '1px solid #E2E8F0',
        'font-weight': '500',
        'font-size': '10px',
        'height': '24px',
        'width': '60%',
        'display': 'inline-block',
        'vertical-align': 'middle'
    },
    'play_button': {
        'width': '48px',
        'height': '48px',
        'border-radius': '50%',
        'background': '#4267B2',
        'border': 'none',
        'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.1)',
        'cursor': 'pointer',
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'color': 'white',
        'font-size': '24px',
        'transition': 'all 0.3s ease',
        'padding': '0'
    },
    'logo_w': {
        'color': '#5D7A5D',  # Gray-green color
    },
    'logo_l': {
        'color': '#A07777',  # Gray-red color
    },
}


def exponential_smoothing(values, alpha=0.6):
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1.")

    values = np.asarray(values, dtype=float)
    smoothed_values = []
    for idx, val in enumerate(values):
        if idx == 0:
            smoothed_values.append(val)
        else:
            smoothed_val = val * alpha + (1 - alpha) * smoothed_values[idx - 1]
            smoothed_values.append(smoothed_val)
    return smoothed_values


@dataclass
class PlotPoint:
    x: float | None
    y: float | None


class UIState:
    """
        A class to represent the state of the UI and all the objects and 
        their properties necessary to be maintained for the UI to function.
    """
    def __init__(self, root_directory: str):
        self.root_directory = root_directory
        self.dirty = False
        self.dirty_dfs = []

        # Details about the hyperparameters
        self.hyperparam = pd.DataFrame(columns=_HYPERPARAM_COLUMNS)
        # Details about the layers
        self.layers_df = pd.DataFrame(columns=_LAYER_DF_COLUMNS)
        # Details about neurons 
        self.neurons_df = pd.DataFrame(columns=_NEURONS_DF_COLUMNS)
        # Details about the metrics
        self.metrics_df = pd.DataFrame(columns=_METRICS_DF_COLUMNS)
        # Details about the annotations
        self.annotation = pd.DataFrame(columns=_ANNOTATIONS_DF_COLUMNS)
        # Details about the data
        self.samples_df = pd.DataFrame(columns=_SAMPLES_DF_COLUMNS)
        # Details about the eval data
        self.eval_samples_df = pd.DataFrame(columns=_SAMPLES_DF_COLUMNS)
        self.task_type = 'classification'


        self.metrics_df_path = os.path.join(
            self.root_directory, "statuses_df.csv")
        self.annotation_path = os.path.join(
            self.root_directory, "annotations.csv")

        if os.path.exists(self.metrics_df_path):
            self.metrics_df = pd.read_csv(self.metrics_df_path)
        if os.path.exists(self.annotation_path):
            self.annotation = pd.read_csv(self.annotation_path)

        # These amount are in vw
        self.layer_div_width_per_col = 4.5
        self.layer_div_width_minimum = 20
        self.layer_id_to_df_row_idx = {}  # layer_id -> idx
        self.neuron_id_to_df_row_idx = {}  # (layer_id, neuron_id) -> idx

        self.selected_neurons = defaultdict(lambda: [])  # layer_id -> List[neuron_id]
        self.lock = threading.RLock()
        self.metrics_lock = threading.RLock()

        # Cache plots for faster rendering
        self.exp_name_2_color = defaultdict(lambda: "blue")
        self.exp_name_metric_name_2_plot = defaultdict(lambda: None)
        self.exp_name_metric_name_2_anot = defaultdict(lambda: [])
        self.exp_name_2_need_redraw = defaultdict(lambda: False)
        self.current_run_id = 0

        self.exp_names = set()
        self.met_names = set()
        self.ant_names = set()

        self.exp_name_metric_name_annot_name_2_values = defaultdict(list)
        if not self.metrics_df.empty:
            self.exp_names = set(self.metrics_df['experiment_name'].unique())
            self.met_names = set(self.metrics_df['metric_name'].unique())
            for idx, exp_name in enumerate(self.exp_names):
                self.exp_name_2_color[exp_name] = _PLOTS_COLOR_WHEEL[
                    idx % len(_PLOTS_COLOR_WHEEL)]
        if not self.annotation.empty:
            self.ant_names = set(self.annotation['annotation'].unique())

        self.plot_name_2_selection_point = defaultdict(lambda: None)
        self.plot_name_2_curr_head_point = defaultdict(lambda: None)
        if not self.metrics_df.empty:
            for met_name in self.met_names:
                relevant_df = self.metrics_df.query(
                    f"metric_name == '{met_name}'")
                if not relevant_df.empty:
                    self.plot_name_2_curr_head_point[met_name] = PlotPoint(
                        relevant_df.iloc[-1]["model_age"],
                        relevant_df.iloc[-1]["metric_value"])

    def get_neurons_df(self):
        with self.lock:
            return self.neurons_df.copy()

    def get_layers_df(self):
        with self.lock:
            return self.layers_df

    def get_plots_for_exp_name_metric_name(
            self, metric_name, exp_name):

        key = (metric_name, exp_name)
        if self.exp_name_metric_name_2_plot[key] and \
                not self.exp_name_2_need_redraw[exp_name]:
            return [self.exp_name_metric_name_2_plot[key]] + \
                self.exp_name_metric_name_2_anot[key]

        with self.metrics_lock:
            relevant_df = self.metrics_df.query(
                f"metric_name == '{metric_name}' and "
                f"experiment_name == '{exp_name}'")

        if relevant_df.empty:
            return []
        first = True
        if 'run_id' in relevant_df.columns:
            traces = []
            for run_id, run_df in relevant_df.groupby("run_id"):
                trace = go.Scatter(
                    x=run_df["model_age"],
                    y=run_df["metric_value"],
                    mode='lines',
                    name=exp_name,
                    line=dict(color=self.exp_name_2_color[exp_name]),
                    showlegend=first,
                )
                traces.append(trace)
                first = False
        else:
            # fallback: original behavior
            traces = [go.Scatter(
                x=relevant_df["model_age"],
                y=relevant_df["metric_value"],
                mode='lines',
                name=exp_name,
                line=dict(color=self.exp_name_2_color[exp_name]),
            )]

        annotation_plots = self._get_annot_plots(exp_name, metric_name)
        return traces + annotation_plots


    def _get_annot_plots(self, exp_name, met_name):
        plots = []
        for anot_name in self.ant_names:
            with self.metrics_lock:
                relevant_df = self.annotation.query(
                    f"experiment_name == '{exp_name}' and "
                    f"annotation == '{anot_name}'")
            if relevant_df.empty:
                continue

            metric_name_2_annot_values = defaultdict(list)
            for _, row in relevant_df.iterrows():
                metadadata = ast.literal_eval(row["metadata"])
                for k, v in metadadata.items():
                    metric_name_2_annot_values[k].append(v)

            # Translate annotation name to metadata key
            annotation_name_in_keys = anot_name
            for key in metric_name_2_annot_values:
                if anot_name in key:
                    annotation_name_in_keys = key
                    break

            if met_name not in metric_name_2_annot_values:
                continue
            if not metric_name_2_annot_values[met_name]:
                continue
            anot = go.Scatter(
                x=relevant_df["model_age"],
                y=metric_name_2_annot_values[met_name],
                mode='markers',
                marker_symbol="diamond",
                name=f"ckpt-{exp_name}",
                customdata=metric_name_2_annot_values[annotation_name_in_keys],
                marker={
                    "color": self.exp_name_2_color[exp_name],
                    "size": 10,
                }
            )
            plots.append(anot)
        return plots

    def __repr__(self) -> str:
        return f"UIState[{self.root_directory}][dirty={self.dirty}]" + \
            f"({len(self.hyperparam)} hyper-parameters, " + \
            f"{len(self.layers_df)} monitored layers, " + \
            f"{len(self.neurons_df)} monitored neurons, " + \
            f"{len(self.metrics_df)} metrics values, " + \
            f"{len(self.annotation)} annotations, " + \
            f"{len(self.samples_df)} monitored samples, " + \
            f"{self.exp_names} " + \
            f"{self.met_names} " + \
            f"{self.ant_names} " + \
            ")"

    def update_from_server_state(
        self, server_state: pb2.CommandResponse):
        """
            Update the UI state with the new state from the server.
        """
        # print("[UIState] Updating from server state.")
        self.update_hyperparams_from_server(
            server_state.hyper_parameters_descs)
        self.update_neurons_from_server(
            server_state.layer_representations)
        self.update_samples_from_server(
            server_state.sample_statistics)

    def update_hyperparams_from_server(
            self, hyper_parameters_descs: List[pb2.HyperParameterDesc]):
        hyper_parameters_descs.sort(key=lambda x: x.name)
        hidx = -1
        for hidx, hyper_parameter_desc in enumerate(hyper_parameters_descs):
            if hyper_parameter_desc.type == "number":
                self.hyperparam.loc[hidx] = [
                    hyper_parameter_desc.label,
                    hyper_parameter_desc.type,
                    hyper_parameter_desc.name,
                    hyper_parameter_desc.numerical_value
                ]
            else:
                self.hyperparam.loc[hidx] = [
                    hyper_parameter_desc.label,
                    hyper_parameter_desc.type,
                    hyper_parameter_desc.name,
                    hyper_parameter_desc.string_value
                ]
        self.hyperparam.loc[hidx + 1] = [
            "Play/Pause Train",
            "button",
            "play_pause",
            False, # is_training
        ]

    def layer_representation_to_df_row(self, layer_representation):
        layer_row = [
            layer_representation.layer_id,
            layer_representation.layer_type,
            layer_representation.layer_name,
            layer_representation.neurons_count,
            layer_representation.incoming_neurons_count,
            layer_representation.kernel_size,
            None,
        ]
        return layer_row

    def neuron_statistics_to_df_row_v2(self, neuron_stats):
        adiff = abs(
                neuron_stats.train_trigger_rate - \
                neuron_stats.eval_trigger_rate)
        rdiff = 0
        if neuron_stats.train_trigger_rate > 0:
            rdiff = adiff / neuron_stats.train_trigger_rate
        neuron_row = [
            -1,
            neuron_stats.neuron_id.neuron_id,
            neuron_stats.neuron_age,
            neuron_stats.train_trigger_rate,
            neuron_stats.eval_trigger_rate,
            adiff,
            rdiff,
            neuron_stats.learning_rate == 0,
            get_neuron_status(neuron_stats).value,
            '',
            False,  # highlighted
        ]
        return neuron_row

    def neuron_statistics_to_df_row(self, layer_representation, neuron_stats):
        adiff = abs(
                neuron_stats.train_trigger_rate - \
                neuron_stats.eval_trigger_rate)
        rdiff = 0
        if neuron_stats.train_trigger_rate > 0:
            rdiff = adiff / neuron_stats.train_trigger_rate
        neuron_row = [
            layer_representation.layer_id,
            neuron_stats.neuron_id.neuron_id,
            neuron_stats.neuron_age,
            neuron_stats.train_trigger_rate,
            neuron_stats.eval_trigger_rate,
            adiff,
            rdiff,
            neuron_stats.learning_rate == 0,
            get_neuron_status(neuron_stats).value,
            layer_representation.layer_type,
            False,  # highlighted
        ]
        return neuron_row

    def update_neurons_from_server(
            self, layer_representations: List[pb2.LayerRepresentation]):
        """
            Update the neurons dataframe with the new neurons details.
        """
        if not layer_representations:
            return
        neuron_row_idx = 0
        neurons_df = pd.DataFrame(columns=_NEURONS_DF_COLUMNS)
        layers_df = pd.DataFrame(columns=_LAYER_DF_COLUMNS)

        for lyr_idx, layer_representation in enumerate(layer_representations):
            layer_row = self.layer_representation_to_df_row(
                layer_representation)
            layers_df.loc[lyr_idx] = layer_row
            self.layer_id_to_df_row_idx[
                layer_representation.layer_id] = lyr_idx

            for neuron_stats in layer_representation.neurons_statistics:
                neuron_row = self.neuron_statistics_to_df_row(
                    layer_representation, neuron_stats)
                neurons_df.loc[neuron_row_idx] = neuron_row
                neuron_row_idx += 1

        neurons_df.set_index(["layer_id", "neuron_id"], inplace=True)

        with self.lock:
            self.neurons_df = neurons_df
            self.layers_df = layers_df

    def update_metrics_from_server(
            self, status: pb2.TrainingStatusEx):
        # print(f"UIState.update_metrics_from_server: {status}")
        self.exp_names.add(status.experiment_name)
        self.exp_name_2_need_redraw[status.experiment_name] = True

        if status.HasField("metrics_status"):
            metrics_row = [
                status.experiment_name,
                status.model_age,
                status.metrics_status.name,
                status.metrics_status.value,
                self.current_run_id
            ]
            with self.metrics_lock:
                self.metrics_df.loc[len(self.metrics_df)] = metrics_row

            self.plot_name_2_curr_head_point[status.metrics_status.name] = \
                PlotPoint(status.model_age, status.metrics_status.value)

            if len(self.metrics_df) % 1000 == 999:
                with self.metrics_lock:
                    self.metrics_df.to_csv(self.metrics_df_path, index=False)
            self.met_names.add(status.metrics_status.name)

            if status.experiment_name not in self.exp_name_2_color:
                self.exp_name_2_color[status.experiment_name] = \
                    _PLOTS_COLOR_WHEEL[ \
                        len(self.exp_names) % len(_PLOTS_COLOR_WHEEL)]
        elif status.HasField("annotat_status"):
            other_metrics = {}
            with self.metrics_lock:
                for _, row in self.metrics_df.iloc[::-1].iterrows():
                    if row["experiment_name"] == status.experiment_name and \
                            row['metric_name'] not in other_metrics:
                        other_metrics[row['metric_name']] = row["metric_value"]
                    if len(other_metrics) == len(self.met_names):
                        print(
                            f"other metrics data age: ", row["model_age"],
                            f" anotation age:", status.model_age)
                        break

            metadata = status.annotat_status.metadata
            metadata.update(other_metrics)
            annotation_row = [
                status.experiment_name,
                status.model_age,
                status.annotat_status.name,
                str(metadata),  # TODO: revise this 
            ]
            with self.metrics_lock:
                self.annotation.loc[len(self.annotation)] = annotation_row
                self.annotation.to_csv(self.annotation_path, index=False)
            self.ant_names.add(status.annotat_status.name)

        print("UI.update_metrics_from_server: ", len(self.metrics_df), len(self.annotation), end="\r")

    # updated function to take lists instead of maps

    def update_samples_from_server(self, sample_statistics: pb2.SampleStatistics):
        try:
            rows = []
            extra_keys = set()
            self.task_type = getattr(sample_statistics, 'task_type', self.task_type)
            for record in sample_statistics.records:
                row = {
                    "SampleId": int(record.sample_id),
                    "Target": list(record.sample_label),            
                    "Prediction": list(record.sample_prediction), 
                    "LastLoss": float(record.sample_last_loss),
                    "Encounters": int(record.sample_encounters),
                    "Discarded": bool(record.sample_discarded),
                }
                for field in getattr(record, "extra_fields", []):
                    if field.HasField("float_value"):
                        row[field.name] = field.float_value
                    elif field.HasField("int_value"):
                        row[field.name] = field.int_value
                    elif field.HasField("string_value"):
                        row[field.name] = field.string_value
                    elif field.HasField("bytes_value"):
                        row[field.name] = field.bytes_value
                    elif field.HasField("bool_value"):
                        row[field.name] = field.bool_value
                    else:
                        row[field.name] = None
                    extra_keys.add(field.name)
                rows.append(row)

            all_columns = list(_SAMPLES_DF_COLUMNS) + sorted(extra_keys)

            df = pd.DataFrame(rows)
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None 
            with self.lock:
                if sample_statistics.origin == "train":
                    self.samples_df = df[all_columns]
                elif sample_statistics.origin == "eval":
                    self.eval_samples_df = df[all_columns]

        except Exception as e:
            print("Error processing sample:", e)




    def get_layer_df_row_by_id(self, layer_id: int):
        with self.lock:
            return self.layers_df.loc[self.layer_id_to_df_row_idx[layer_id]]


def get_pause_play_button():
    return dbc.Button(
        id='resume-pause-train-btn',
        children="▶",
        style=custom_styles['play_button'],
        n_clicks=0
    )


def get_label_and_input_row(
    dfrow_param_desc: Dict[str, Any] | None = None
):
    label = dfrow_param_desc["label"]
    ident = dfrow_param_desc["name"]
    type_ = dfrow_param_desc["type"]
    deflt = dfrow_param_desc["value"]
    return html.Div([
        html.Label(label, style=custom_styles['config_label']),
        dbc.Input(
            id={"type": "hyper-params-input", "idx": ident},
            type=type_,
            value=deflt,
            style=custom_styles['input_style']
        )
    ], style=custom_styles['config_card'])


def get_header_hyper_params_div(
        ui_state: UIState | None = None
) -> html.Div:
    children = []
    for idx, row in ui_state.hyperparam.iterrows():
        if row["name"] == "play_pause":
            continue
        children.append(
            dbc.Col([get_label_and_input_row(row)], md=3, lg=2))

    section = html.Div(
        id="hyper-parameters-panel",
        children=[
            dbc.Row(
                id="hyper-params-row",
                children=children,
                style={
                    "textWeight": "bold",
                    "width": "100%",  # Changed from 80vw to 100%
                    "align": "center",
                    'margin': '0 auto',
                    'padding': '5px',
                    'flexWrap': 'wrap'  # Ensure items wrap to next line
                }
            ),
        ],
        style={
            # Removed backgroundColor: "#DDD"
            "width": "100%"  # Ensure full width
        }
    )

    header_div = html.Div([
        html.H1("WeightsLab", style={'textAlign': 'center', 'display': 'inline-block', 'marginRight': '20px'}),
        get_pause_play_button()
    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'})

    return html.Div(
        [header_div, section],
        style=custom_styles['main_card']
    )

class NeuronStatus(str, Enum):
    """Enum for neuron status in order to properly color code them."""
    NEUTRAL = "N/A"  # "neutral"
    OVERFIT = "OVRFT"  # "overfit"
    SUSPECT = "BAD"  # "suspect"
    IMPACT0 = "DEAD"  # "impact0"
    FROZEND = "FROZEN"  # "frozend"
    HEALTHY = "GREAT"  # "healthy"


def get_minus_neurons_button(layer_id):
    button = dbc.Button(
        "-",
        id={"type": "layer-rem-btn", "layer_id": layer_id},
        color='transparent',
        n_clicks=0,
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "red",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_plus_neurons_button(layer_id):
    button = dbc.Button(
        "+",
        id={"type": "layer-add-btn", "layer_id": layer_id},
        color='transparent',
        n_clicks=0,
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "green",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_inspect_neurons_button(layer_id):
    button = dbc.Button(
        "🔍",
        id={"type": "layer-see-btn", "layer_id": layer_id},
        color='transparent',
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "blue",
            'width': _LAYER_BUTTON_WIDTH,
            'height':_LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_freeze_neurons_button(layer_id):
    button = dbc.Button(
        "❄",
        id={"type": "layer-freeze-btn", "layer_id": layer_id},
        color='transparent',
        style={
            'fontSize': _LAYER_BUTTON_FNT_SZ,
            "borderColor": "transparent",
            "color": "blue",
            'width': _LAYER_BUTTON_WIDTH,
            'height': _LAYER_BUTTON_HEIGHT,
        }
    )
    return button


def get_layer_ops_buttons(layer_id):
    button_minus = get_minus_neurons_button(layer_id)
    button_plus = get_plus_neurons_button(layer_id)
    button_freeze = get_freeze_neurons_button(layer_id)
    # button_inspect = get_inspect_neurons_button(layer_id)
    return html.Div(
        dbc.Row(
            dbc.Col(
                [button_minus, button_plus, button_freeze]
            )
        ),
        style={
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "center",
        }
    )


def get_neuron_status(neuron_stats):
    adiff = abs(
        neuron_stats.train_trigger_rate - \
        neuron_stats.eval_trigger_rate)
    rdiff = 0
    if neuron_stats.train_trigger_rate > 0:
        rdiff = adiff / neuron_stats.train_trigger_rate

    if neuron_stats.learning_rate == 0.0:
        return NeuronStatus.FROZEND

    if neuron_stats.neuron_age <= 10000:
        return NeuronStatus.NEUTRAL

    if neuron_stats.train_trigger_rate <= 0.01 or \
            neuron_stats.eval_trigger_rate <= 0.01:
        return NeuronStatus.IMPACT0

    if rdiff >= .50:
        return NeuronStatus.OVERFIT
    elif rdiff >= .10:
        return NeuronStatus.SUSPECT

    return NeuronStatus.HEALTHY


def get_neuron_status_color(neuron_status: NeuronStatus) -> str:
    if neuron_status == NeuronStatus.NEUTRAL:
        return "gray"
    elif neuron_status == NeuronStatus.OVERFIT:
        return "red"
    elif neuron_status == NeuronStatus.SUSPECT:
        return "orange"
    elif neuron_status == NeuronStatus.IMPACT0:
        return "black"
    elif neuron_status == NeuronStatus.FROZEND:
        return "blue"
    elif neuron_status == NeuronStatus.HEALTHY:
        return "green"
    else:
        return "PURPLE"


def format_large_int(n, max_length=5):
    suffix = ''
    if abs(n) >= 10**12:
        n, suffix = n / 10**12, 'T'
    elif abs(n) >= 10**9:
        n, suffix = n / 10**9, 'B'
    elif abs(n) >= 10**6:
        n, suffix = n / 10**6, 'M'
    elif abs(n) >= 10**3:
        n, suffix = n / 10**3, 'K'
    else:
        return int(n)

    formatted_number = f"{n:.1f}{suffix}"
    if len(formatted_number) > max_length:
        formatted_number = f"{int(n)}{suffix}"

    if len(formatted_number) > max_length:
        formatted_number = f"{n:.0f}{suffix}"

    return formatted_number


def format_value(value):
    if isinstance(value, float):
        return round(value, 2)
    return value


def get_neuron_stats_div(neuron_row, checklist_values):
    COL_MAX_LEN = 5
    row = []
    for checklist_value in checklist_values:
        value_str = ""
        if checklist_value == 'neuron_age':
            value_str = format_large_int(neuron_row['Age'])
        elif checklist_value == 'trigger_rate_train':
            value_str = "%05.4f" % neuron_row['RTrn']
        elif checklist_value == 'trigger_rate_eval':
            value_str = "%05.4f" % neuron_row['REval']
        elif checklist_value == 'abs_diff':
            value_str = "%05.4f" % neuron_row['ADiff']
        elif checklist_value == 'rel_diff':
            value_str = "%05.4f" % neuron_row['RDiff']

        if value_str:
            col = dbc.Col(value_str[:COL_MAX_LEN])
            row.append(col)

    for checklist_value in checklist_values:
        if checklist_value == 'frozen':
            freeze_button = daq.BooleanSwitch(
                id={
                    'type': 'neuron-frozen-switch',
                    'layer': neuron_row['layer_id'],
                    'neuron': neuron_row['neuron_id'],
                },
                on=neuron_row['Frozen'],
            )
            col = dbc.Col(freeze_button)
            row.append(col)
        if checklist_value == 'status':
            color = get_neuron_status_color(neuron_row['Status'])
            col = dbc.Col(
                neuron_row['Status'][:COL_MAX_LEN],
                style={"color": color, }
            )
            row.append(col)

    bkg_color = '#DDD' if not neuron_row['highlighted'] else "#FFD700",
    return dbc.Row(
        id=str((neuron_row['layer_id'], neuron_row['neuron_id'])),
        children=row,
        style={
            "color": "black",
            "align": "center",
            "fontWeight": "bold",
            "fontFamily": "Monospace",
            'backgroundColor': bkg_color,
            'margin': '3px',
            'borderRadius': '10px',
            'minWidth': "12vw",
            'padding': '2px',
            'fontSize': '12px',
        }
    )


def get_layer_headings(layer_row) -> html.Div:
    heading = layer_row.layer_type
    heading += f"[id={layer_row.layer_id}]"
    sub_heading = ""

    if layer_row.layer_type == "Conv2d":
        sub_heading += f" [{layer_row.incoming}->"
        sub_heading += f"{layer_row.kernel_size}x"
        sub_heading += f"{layer_row.kernel_size}->"
        sub_heading += f"{layer_row.outgoing}]"
    if layer_row.layer_type == "Linear":
        sub_heading += f" [{layer_row.incoming}->"
        sub_heading += f"{layer_row.outgoing}]"

    return heading, sub_heading


def get_neuron_query_input_div(ui_state: UIState):
    del ui_state
    cols = []

    cols.append(
        dbc.Col(
            dbc.Input(
                id='neuron-query-input', type='text',
                placeholder='Enter query predicate to select neurons.',
                style={'width': '18vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Input(
                id='neuron-query-input-weight', type='number',
                placeholder='weight',
                style={'width': '6vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dcc.Dropdown(
                id="neuron-action-dropdown",
                options=[
                    {"label": "Highlight", "value": "highlight"},
                    {"label": "Delete", "value": "delete"},
                    {"label": "Reinitialize", "value": "reinitialize"},
                    {"label": "Freeze", "value": "freeze"},
                    {"label": "Add Neurons", "value": "add_neurons"}
                ],
                value="highlight",  # Default value
                placeholder="Select an action",  # Placeholder text
                style={'width': '8vw'}  # Style to control dropdown width
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Button(
                "Run", id='run-neuron-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        )
    )

    neuron_quering_row = dbc.Row(
        cols,
        style={
            "display": 'flex',
            "justifyContent": 'center',
            "alignItems": 'center',
            "align": "center",
            "width": "45vw",
        }
    )
    return dbc.Col(
        neuron_quering_row,
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
        }
    )


def convert_checklist_to_df_head(checklist_values):
    heading = []
    for checklist_value in checklist_values:
        if checklist_value == 'neuron_id':
            heading.append("neuron_id")
        if checklist_value == 'neuron_age':
            heading.append("Age")
        if checklist_value == 'trigger_rate_train':
            heading.append("RTrn")
        if checklist_value == 'trigger_rate_eval':
            heading.append("REval")
        if checklist_value == 'abs_diff':
            heading.append("ADiff")
        if checklist_value == 'rel_diff':
            heading.append("RDiff")
        if checklist_value == 'weight_diff':
            heading.append("WDiff")
        if checklist_value == 'bias_diff':
            heading.append("BDiff")
    for checklist_value in checklist_values:
        if checklist_value == 'frozen':
            heading.append("Frzn")
        if checklist_value == 'status':
            heading.append("Status")

    return heading


def format_values_df(df):
    formated_df = pd.DataFrame(columns=df.columns)
    for col in df.columns:  # Skip "Neuron" column
        if col == "neuron_id" or col == "Age":
            formated_df[col] = df[col].apply(format_large_int).astype(str)
        else:
            formated_df[col] = df[col].apply(format_value)
    return formated_df


def layer_div_width(checklist_values):
    layer_width = LYR_BASE_WIDTH + len(checklist_values) * WIDTH_PER_COLUMN
    return layer_width


def get_layer_div(
        layer_row,
        layer_neurons_df,
        ui_state,
        checklist_values,
):
    heading, sub_heading = get_layer_headings(layer_row)
    layer_width = layer_div_width(checklist_values)

    checklist_values = convert_checklist_to_df_head(checklist_values)
    try:
        neurons_view_df = format_values_df(layer_neurons_df[checklist_values])
    except Exception as e:
        return no_update

    return html.Div(
        id={"type": "layer-div", "layer_id": layer_row['layer_id']},
        children=[
            html.H3(heading, style={'textAlign': 'center'}),
            html.H4(
                id={"type": "layer-sub-heading", "layer_id": layer_row['layer_id']},
                children=sub_heading, style={'textAlign': 'center'}),
            dash_table.DataTable(
                id={"type": "layer-data-table", "layer_id": layer_row['layer_id']},
                data=neurons_view_df.to_dict("records"),
                columns=[
                    {"name": col, "id": col}
                    for col in neurons_view_df.columns],
                sort_action="native",
                row_selectable='single',
                style_table={
                    "maxHeight": "300px",
                    "overflowY": "scroll"
                },
                style_cell={
                    "minWidth": "60px",
                    "width": "60px",
                    "maxWidth": f"{WIDTH_PER_COLUMN}px"
                },
                style_data={"whiteSpace": "normal", "height": "auto"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{Status} = 'OVRFT'"},
                        "color": "red",  # Text color for 'OVERFIT'
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": "{Status} = 'BAD'"},
                        "color": "orange",  # Text color for 'BAD'
                    },
                    {
                        "if": {"filter_query": "{Status} = 'GREAT'"},
                        "color": "green",  # Text color for 'GREAT'
                        "fontWeight": "bold",
                    },
                    {
                        "if": {"filter_query": "{Status} = 'DEAD'"},
                        "color": "black",  # Text color for 'DEAD'
                        "backgroundColor": "#e0e0e0",  # Light gray background for DEAD
                    },
                    {
                        "if": {"filter_query": "{Status} = 'FROZEN'"},
                        "color": "#B7C9E2",  # Text color for 'FROZEN'
                        # "backgroundColor": "#B7C9E2"
                    },
                    {
                        "if": {"filter_query": "{Status} = 'N/A'"},
                        "color": "#a0a0a0",  # Text color for 'N/A'
                        "fontStyle": "italic",
                    },
                ],
            ),
            get_layer_ops_buttons(layer_row['layer_id']),
        ],
        style={
            "minWidth": f"{layer_width}px",
            "padding": "10px",
            "margin": "5px",
            'border': '2px solid #666',
            'borderRadius': '15px',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        },
    )


def interactable_layers(
    ui_state: UIState,
    checklist_values : Tuple[str] = _DEFAULT_CHECKLIST_VALUES,
):
    children = []
    for _, layer_row in ui_state.layers_df.iterrows():
        try:
            layer_neurons_df = ui_state.get_neurons_df().loc[
                layer_row.layer_id].copy()
            layer_neurons_df = layer_neurons_df.reset_index()
            layer_neurons_df['layer_id'] = layer_row['layer_id']

            children.append(
                get_layer_div(
                    layer_row, layer_neurons_df, ui_state, checklist_values))
        except Exception as e:
            print("neurons_df: ", ui_state.get_neurons_df().head())
            print(f"Error getting layer data: {layer_row['layer_id']}: {e}")
            continue

    return html.Div(
        id="layer-weights",
        children=children,
        style={
            'display': 'flex',
            'overflowX': 'scroll',
            "padding": "10px",
            "border": "1px solid #ccc",
        }
    )


def stats_display_checklist(ui_state: UIState):
    del ui_state
    checklist = dcc.Checklist(
        id='neuron_stats-checkboxes',
        options=[
            {'label': 'Neuron Id', 'value': 'neuron_id'},
            {'label': 'Neuron Age', 'value': 'neuron_age'},
            {'label': 'Train Set Trigger Rate', 'value': 'trigger_rate_train'},
            {'label': 'Eval Set Trigger Rate', 'value': 'trigger_rate_eval'},
            {'label': 'Absolute Difference between rates', 'value': 'abs_diff'},
            {'label': 'Relative Difference between rates', 'value': 'rel_diff'},
            {'label': 'Health State', 'value': 'status'},
        ],
        value=['neuron_id', 'neuron_age', 'trigger_rate_train', 'status'],
        inline=True,
        labelStyle={'marginRight': '10px'},
    )
    return dbc.Col(
        children=[checklist],
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
        }
    )


def get_data_query_input_div(ui_state: UIState):
    cols = []

    cols.append(
        dbc.Col(
            dbc.Input(
                id='train-data-query-input', type='text',
                placeholder='Enter train data query',
                style={'width': '18vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Input(
                id='data-query-input-weight', type='number',
                placeholder='weight',
                style={'width': '4vw'}
            ),
        )
    )
    cols.append(
        dbc.Col(
            dbc.Button(
                "Run", id='run-train-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        )
    )

    return dbc.Row(cols)


def get_weights_modal(ui_state: UIState):
    return dbc.Modal(
        [
            dbc.ModalHeader("Edit Weights"),
            dbc.ModalBody([
                dcc.Graph(id='weight-heatmap', config={'displayModeBar': False}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Apply Changes", id="apply-weights", color="primary", className="ml-auto")
            ),
        ],
        id="modal-weights-edit",
        size="lg",  # Make the modal large
        is_open=False,  # Initially closed
    )

def zerofy_checklist():
    checklist = dcc.Checklist(
        id='zerofy-options-checklist',
        options=[
            {'label': 'zerofy with frozen', 'value': 'frozen'},
            {'label': 'zerofy with older',  'value': 'older'},
        ],
        value=[],         
        inline=True,
        labelStyle={'marginRight': '10px'},
    )
    return dbc.Col(
        children=[checklist],
        style={
            'display': 'flex',
            'justifyContent': 'center',
            'alignItems': 'center',
        }
    )



def get_weights_div(ui_state: UIState):
    return html.Div(
        id="model-architecture-div",
        children=[
            stats_display_checklist(ui_state),
            zerofy_checklist(),
            interactable_layers(ui_state),
            get_neuron_query_input_div(ui_state),
            get_weights_modal(ui_state),
        ],
    )


def get_plots_div():
    experiment_checkboxes = dcc.Checklist(
        id='experiment_checklist',
        options=[],
        value=[],
        labelStyle={
            'display': 'block'
        },
        style={
            'overflowY': 'auto',
            "margin": "8px",
        }
    )

    experiment_management = dbc.Col(
        id="experiment_management",
        children=[
            # experiment_smoothness,
            # experiment_smooth_sld,
            experiment_checkboxes,
        ],
        width=1,
    )

    experiment_plots_div = dbc.Col(
        id="experiment_plots_div",
        children=[
            #dcc.Graph(id='experiment-plot'),
        ],
        width=11,
    )

    return dbc.Row(
        id="plots-panel",
        children=[
            experiment_management,
            experiment_plots_div,
        ],
        style={
            'marginTop': '20px',  # Add top margin to create space
            'paddingTop': '10px'  # Add top padding for additional space
        }
    )


def sample_statistics_to_data_records(
        sample_statistics: pb2.SampleStatistics
):
    data_records = []
    for sample_id in range(sample_statistics.sample_count):
        data_record = {
            "SampleId": sample_id,
            "Label": sample_statistics.sample_label[sample_id],
            "Prediction": sample_statistics.sample_prediction[sample_id],
            "LastLoss": sample_statistics.sample_last_loss[sample_id],
            "Encounters": sample_statistics.sample_encounters[sample_id],
            "Discarded": sample_statistics.sample_discarded[sample_id]
        }
        data_records.append(data_record)
    return data_records


def get_data_tab(ui_state: UIState):
    cols = []
    for column in _DISPLAY_COLUMNS:
        spec = {"name": column, "id": column,
                "type": "text" if column == "Discarded" else 'any'}
        if column == "LastLoss":
            spec["type"] = "numeric"
            spec["format"] = Format(precision=2, scheme=Scheme.fixed)
        cols.append(spec)

    grid_preset_dropdown = dcc.Dropdown(
        id='grid-preset-dropdown',
        options=[
            {'label': str(x * x), 'value': x * x} for x in [3, 4, 5, 6, 7, 10]
        ],
        value=9,
        clearable=False,
        style={'width': '6vw'}
    )

    eval_grid_dropdown = dcc.Dropdown(
        id='eval-grid-preset-dropdown',
        options=[{'label': str(x * x), 'value': x * x} for x in [3, 4, 5, 6, 7, 10]],
        value=9,
        clearable=False,
        style={'width': '6vw'}
    )

    train_table = dash_table.DataTable(
        id='train-data-table',
        data=ui_state.samples_df.to_dict('records'),
        columns=cols,
        sort_action="native",
        page_action="native",
        page_size=16,
        row_selectable='multi',
        style_data_conditional=[],
        row_deletable=True,
        editable=True,
        virtualization=True,
        style_table={
            'height': 'auto',
            'overflowY': 'auto',
            'width': 'auto',
            'margin': '2px',
            'padding': '2px'
        },
        style_cell={'textAlign': 'left', 'minWidth': '4vw', 'maxWidth': '4.5vw'}
    )

    eval_table = dash_table.DataTable(
        id='eval-data-table',
        data=ui_state.eval_samples_df.to_dict('records'),
        columns=cols,
        sort_action="native",
        page_action="native",
        page_size=16,
        row_selectable='multi',
        style_data_conditional=[],
        row_deletable=True,
        editable=True,
        virtualization=True,
        style_table={
            'height': 'auto',
            'overflowY': 'auto',
            'width': 'auto',
            'margin': '2px',
            'padding': '2px'
        },
        style_cell={'textAlign': 'left', 'minWidth': '4vw', 'maxWidth': '4.5vw'}
    )

    train_controls = html.Div([
        dcc.Checklist(
            id='table-refresh-checkbox',
            options=[
                {'label': 'Refresh regularly', 'value': 'refresh_regularly'},
                {'label': 'Discard by flag flip', 'value': 'discard_by_flag_flip'}
            ],
            value=['refresh_regularly', 'discard_by_flag_flip'],
            inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        dcc.Checklist(
            id='sample-inspect-checkboxes',
            options=[{'label': 'Inspect on click', 'value': 'inspect_sample_on_click'}],
            value=[], inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        html.Div(grid_preset_dropdown, style={'marginLeft': '1vw'})
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1vw'})

    eval_controls = html.Div([
        dcc.Checklist(
            id='eval-table-refresh-checkbox',
            options=[
                {'label': 'Refresh regularly', 'value': 'refresh_regularly'},
                {'label': 'Discard by flag flip', 'value': 'discard_by_flag_flip'}
            ],
            value=['refresh_regularly', 'discard_by_flag_flip'],
            inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        dcc.Checklist(
            id='eval-sample-inspect-checkboxes',
            options=[{'label': 'Inspect on click', 'value': 'inspect_sample_on_click'}],
            value=[], inline=True,
            labelStyle={'marginRight': '5px'}
        ),
        html.Div(eval_grid_dropdown, style={'marginLeft': '1vw'})
    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1vw'})


    train_query_div = dbc.Row([
        dbc.Col(
            dbc.Input(
                id='train-data-query-input', type='text',
                placeholder='Enter train data query',
                style={'width': '18vw'}
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='train-query-discard-toggle',
                options=[{'label': 'Un-discard', 'value': 'undiscard'}],
                value=[],
                inline=True
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='train-denylist-accumulate-checkbox',
                options=[{'label': 'Accumulate', 'value': 'accumulate'}],
                value=[], 
                inline=True,
                style={'marginLeft': '1vw'}
            ),
        ),
        dbc.Col(
            dbc.Input(
                id='data-query-input-weight', type='number',
                placeholder='weight',
                style={'width': '4vw'}
            ),
        ),
        dbc.Col(
            dbc.Button(
                "Run", id='run-train-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        ),
    ])

    eval_query_div = dbc.Row([
        dbc.Col(
            dbc.Input(
                id='eval-data-query-input', type='text',
                placeholder='Enter eval data query',
                style={'width': '18vw'}
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='eval-query-discard-toggle',
                options=[{'label': 'Un-discard', 'value': 'undiscard'}],
                value=[],
                inline=True
            ),
        ),
        dbc.Col(
            dbc.Checklist(
                id='eval-denylist-accumulate-checkbox',
                options=[{'label': 'Accumulate', 'value': 'accumulate'}],
                value=[],
                inline=True,
                style={'marginLeft': '1vw'}
            ),
        ),
        dbc.Col(
            dbc.Input(
                id='eval-data-query-weight', type='number',
                placeholder='weight',
                style={'width': '4vw'}
            ),
        ),
        dbc.Col(
            dbc.Button(
                "Run", id='run-eval-data-query', color='primary',
                n_clicks=0,
                style={'width': '3vw'}
            ),
        ),
    ])

    tabs = dcc.Tabs(
        id='data-tabs',
        value='train',
        children=[
            dcc.Tab(label='Train Dataset', value='train', children=[
                html.Div([
                    html.H2("Train Dataset", id="train-dataset-header"),
                    train_controls,
                    html.Div([
                        html.Div([train_table], style={
                            'flex': '0 0 35vw', 
                            'minWidth': '35vw',
                            'height': '100%'
                        }),
                        html.Div([
                            html.Div(id='train-sample-panel')
                        ], style={
                            'flex': '1', 
                            'minWidth': '400px', 
                            'height': 'auto',  
                            'display': 'flex',
                            'overflow': 'auto',
                            'alignItems': 'flex-start',
                            'justifyContent': 'center'
                        })
                    ], style={
                        'display': 'flex', 
                        'alignItems': 'stretch',
                        'gap': '1vw',
                        'width': '100%'
                    }),
                    train_query_div
                ])

            ]),
            dcc.Tab(label='Eval Dataset', value='eval', children=[
                html.Div([
                    html.H2("Eval Dataset"),
                    eval_controls,
                    html.Div([
                        html.Div([eval_table], style={
                            'flex': '0 0 35vw',  
                            'minWidth': '35vw',
                            'height': '100%'

                        }),
                        html.Div([
                            html.Div(id='eval-sample-panel')
                        ], style={
                            'flex': '1', 
                            'minWidth': '400px', 
                            'height': 'auto', 
                            'overflow': 'auto', 
                            'display': 'flex',
                            'alignItems': 'flex-start',
                            'justifyContent': 'center'
                        })
                    ], style={
                        'display': 'flex', 
                        'alignItems': 'stretch',
                        'gap': '1vw',
                        'width': '100%'
                    }),
                    eval_query_div
                ])

            ])
        ]
    )

    return html.Div(tabs, style={
        'margin': '4vw', 'padding': '2vw',
        'borderRadius': '15px', 'border': '2px solid #666',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
        'width': '87vw'
    })

def label_below_img(img_component, last_loss, img_size):
    return html.Div([
        img_component,
        html.Div(
            f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -",
            style={'fontSize': '11px', 'lineHeight': '15px', 'textAlign': 'center', 'marginTop': '2px'}
        )
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': f'{img_size}px'})


def render_segmentation_triplet(input_b64, gt_mask_b64, pred_mask_b64, is_selected, img_size, is_discarded, sid=None, last_loss=None):
    input_img_div = html.Div([
        html.Img(
            src=f'data:image/png;base64,{input_b64}',
            style={'width':f'{img_size}px','border':'1px solid #888'}
        ),
        html.Div(
            f"ID: {sid}", style={
                'position': 'absolute', 'top': '2px', 'left': '4px',
                'background': 'rgba(0,0,0,0.55)', 'color': 'white',
                'fontSize': '10px', 'padding': '1px 5px', 'borderRadius': '3px'
            }
        ) if sid is not None else None,
        html.Div(
            f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -",
            style={
                'position': 'absolute', 'top': '2px', 'right': '4px',
                'background': 'rgba(0,0,0,0.55)', 'color': 'white',
                'fontSize': '10px', 'padding': '1px 5px', 'borderRadius': '3px'
            }
        ) if last_loss is not None else None,
        html.Div("Input", style={'fontSize':10, 'textAlign':'center'})
    ], style={'position': 'relative', 'display': 'inline-block', 'width': f'{img_size}px', 'height': f'{img_size}px'})
    
    return html.Div([
        input_img_div,
        html.Div([
            html.Img(src=f'data:image/png;base64,{gt_mask_b64}', style={'width':f'{img_size}px','border':'1px solid green'}),
            html.Div("Target", style={'fontSize':10, 'textAlign':'center'})
        ]),
        html.Div([
            html.Img(src=f'data:image/png;base64,{pred_mask_b64}', style={'width':f'{img_size}px','border':'1px solid blue'}),
            html.Div("Prediction", style={'fontSize':10, 'textAlign':'center'})
        ]),
    ], style={
        'display':'flex', 
        'flexDirection':'row', 
        'gap':'4px', 
        'marginBottom':'8px', 
        'border': '4px solid red' if is_selected else 'none',
        'transition': 'border 0.3s, opacity 0.3s',
        'opacity': 0.25 if is_discarded else 1.0  
    })

def render_images(ui_state: UIState, stub, sample_ids, selected_ids, origin, discarded_ids=None):
    task_type = getattr(ui_state, "task_type", "classification")
    imgs = []
    num_images = len(sample_ids)
    cols = isqrt(num_images) or 1
    rows = cols
    img_size = int(512 / max(cols, rows))
    if origin == "train":
        df = ui_state.samples_df
    else:
        df = ui_state.eval_samples_df
    id_to_loss = {int(row["SampleId"]): row.get("LastLoss", None) for _, row in df.iterrows()}

    try:
        batch_response = stub.GetSamples(pb2.BatchSampleRequest(
            sample_ids=sample_ids,
            origin=origin,
            resize_width=img_size,
            resize_height=img_size
        ))
        if task_type == "segmentation":
            for sample in batch_response.samples:
                sid = sample.sample_id
                input_b64 = base64.b64encode(sample.raw_data).decode('utf-8')
                gt_mask_b64 = base64.b64encode(sample.mask).decode('utf-8') if sample.mask else ""
                pred_mask_b64 = base64.b64encode(sample.prediction).decode('utf-8') if sample.prediction else ""
                is_selected = sid in selected_ids
                is_discarded = sid in (discarded_ids or set())
                last_loss = id_to_loss.get(sid, None)
                triplet = render_segmentation_triplet(ui_state, stub, input_b64, gt_mask_b64, pred_mask_b64, is_selected, img_size, is_discarded, sid, last_loss)
                triplet_with_label = html.Div([
                    triplet,
                    html.Div(
                        f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -",
                        style={'fontSize': '11px', 'lineHeight': '15px', 'textAlign': 'center', 'marginTop': '2px'}
                    )
                ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': f'{img_size*3 + 12}px'})
                imgs.append(triplet_with_label)


        else:
            for sample in batch_response.samples:
                sid = sample.sample_id
                b64 = base64.b64encode(sample.raw_data).decode('utf-8')
                is_selected = sid in selected_ids
                is_discarded = sid in (discarded_ids or set())
                border = '4px solid red' if is_selected else '1px solid #ccc'
                style = {
                    'width': f'{img_size}px',
                    'height': f'{img_size}px',
                    'margin': '0.1vh',
                    'border': border,
                    'transition': 'border 0.3s, opacity 0.3s',
                    'objectFit': 'contain',
                    'imageRendering': 'auto',
                    'opacity': 0.25 if is_discarded else 1.0  
                }
                img = html.Img(src=f'data:image/png;base64,{b64}', style=style)
                last_loss = id_to_loss.get(sid, None)
                imgs.append(label_below_img(img, last_loss, img_size))


    except Exception as e:
        print(f"[ERROR] {origin} sample rendering failed: {e}")
        return no_update


    return html.Div(children=imgs, style={
        'display': 'grid',
        'gridTemplateColumns': f'repeat({cols}, 1fr)',
        'columnGap': '0.1vw',
        'rowGap': '0.1vh',
        'width': '100%',
        'height': 'auto',
        'maxWidth': 'calc(100vw - 40vw)', 
        'boxSizing': 'border-box',
        'justifyItems': 'center',
        'alignItems': 'center',
        'paddingLeft': '0.01vw'
    })

def parse_sort_info(query):
    if not query or 'sortby' not in query.lower():
        return None
    match = re.search(r'sortby\s+([a-zA-Z0-9_, \s]+)', query, re.IGNORECASE)
    if not match:
        return None
    cols, dirs = [], []
    for part in match.group(1).split(','):
        tokens = part.strip().split()
        if not tokens:
            continue
        col = tokens[0]
        direction = tokens[1].lower() if len(tokens) > 1 and tokens[1].lower() in ['asc', 'desc'] else 'asc'
        cols.append(col)
        dirs.append(direction == 'asc')
    return {'cols': cols, 'dirs': dirs} if cols else None

def format_for_table(val, task_type):
    if val is None:
        return "-"
    if task_type == "segmentation":
        return str(val)
    else:
        if isinstance(val, list):
            try:
                return int(val[0])
            except Exception:
                return str(val)
        return str(val)


def rewrite_query_for_lists(query, task_type, ui_state: UIState):
    if task_type == "segmentation" and query:
        pattern = re.compile(r"(\d+)\s+in\s+(Target|Prediction)")
        matches = pattern.findall(query)
        if matches:
            def filter_fn(df):
                mask = None
                for val, col in matches:
                    this_mask = df[f"{col}ClassesStr"].str.split(",").apply(lambda x: any(xx == val for xx in x))
                    if mask is None:
                        mask = this_mask
                    else:
                        mask &= this_mask
                return df[mask]
            return filter_fn
    return None

def get_query_context(tab_type, ui_state: UIState):
    if tab_type == "train":
        df = ui_state.samples_df.copy()
        remove_op = "remove_from_denylist_operation"
        deny_op = "deny_samples_operation"
    else:
        df = ui_state.eval_samples_df.copy()
        remove_op = "remove_eval_from_denylist_operation"
        deny_op = "deny_eval_samples_operation"
    return df, remove_op, deny_op


def get_ui_app_layout(ui_state: UIState) -> html.Div:
    layout_children = [
        dcc.Store(id='train-image-selected-ids', data=[]),
        dcc.Store(id='eval-image-selected-ids', data=[]),
        dcc.Interval(id='weights-render-freq', interval=1*1000, n_intervals=0),
        dcc.Interval(id='datatbl-render-freq', interval=10*1000, n_intervals=0),
        dcc.Interval(id='graphss-render-freq', interval=10*1000, n_intervals=0),
    ]
    layout_children.append(get_header_hyper_params_div(ui_state))
    layout_children.append(get_plots_div())
    layout_children.append(get_weights_div(ui_state))
    layout_children.append(get_data_tab(ui_state))

    return html.Div(children=layout_children)


def parse_args():
    parser = argparse.ArgumentParser(
        description="WeightsLAb Dash UI")

    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        help="Path to the directory",
        default="./fashion-mnist-dev/"
    )
    args = parser.parse_args()
    if not os.path.isdir(args.root_directory):
        print(f"Error: {args.root_directory} is not a valid directory.")
        sys.exit(1)
    return args


def main():
    args = parse_args()

    channel = grpc.insecure_channel(
        'localhost:50051',
        options=[('grpc.max_receive_message_length', 32 * 1024 * 1024)]
    )
    stub = pb2_grpc.ExperimentServiceStub(channel)
    ui_state = UIState(args.root_directory)
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.ZEPHYR],
        prevent_initial_callbacks='initial_duplicate')

    app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    .config-card:hover {
                        border-color: #4267B2 !important;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 12px rgba(66, 103, 178, 0.3);
                    }
                    .play-pause-btn:hover {
                        transform: scale(1.1);
                        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''

    get_initial_state_request = pb2.TrainerCommand(
        get_hyper_parameters=True,
        get_interactive_layers=True,
        # get_data_records="train",
    )
    print("[UI] About Fetching initial state.")

    with ScopeTimer(tag="initial_state_fetch_and_update") as t:
        initial_state_response = stub.ExperimentCommand(
            get_initial_state_request)
    print(t)
    print("[UI] FetchiED initial state.")
    ui_state.update_from_server_state(initial_state_response)

    print(ui_state)
    app.layout = get_ui_app_layout(ui_state)

    def fetch_server_state_and_update_ui_state():
        while True:
            try:
                for dataset in ["train", "eval"]:
                    req = pb2.TrainerCommand(
                        get_hyper_parameters=(dataset == "train"),
                        get_interactive_layers=(dataset == "train"),
                        get_data_records=dataset
                    )
                    state = stub.ExperimentCommand(req)
                    ui_state.update_from_server_state(state)
            except Exception as e:
                print("Error updating UI state:", e)

    consistency_thread = threading.Thread(
        target=fetch_server_state_and_update_ui_state, daemon=True)
    consistency_thread.start()

    def retrieve_training_statuses():
        nonlocal ui_state, stub
        for status in stub.StreamStatus(pb2.Empty()):
            ui_state.update_metrics_from_server(status)
    status_thread = threading.Thread(
        target=retrieve_training_statuses, daemon=True)
    status_thread.start()

    @app.callback(
        Output('resume-pause-train-btn', 'children', allow_duplicate=True),
        Input({"type": "hyper-params-input", "idx": ALL}, "value"),
        Input('resume-pause-train-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def send_to_controller_hyper_parameters_on_change(
            hyper_param_values, resume_pause_clicks):
        print(
            f"[UI] WeightsLab.send_to_controller_hyper_parameters_on_change"
            f" {hyper_param_values}, {resume_pause_clicks}")
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        button_children = no_update
        prop_id = ctx.triggered[0]['prop_id']
        hyper_parameter = pb2.HyperParameters()

        if "resume-pause-train-btn" in prop_id:
            is_training = resume_pause_clicks % 2
            hyper_parameter.is_training = is_training
            if is_training:
                button_children = "⏸"
                hyper_parameter.training_steps_to_do = hyper_param_values[5]
            else:
                button_children = "▶"
                hyper_parameter.training_steps_to_do = 0
        else:
            btn_dict = eval(prop_id.split('.')[0])
            hyper_parameter_id = btn_dict['idx']

            if hyper_parameter_id == "experiment_name":
                hyper_parameter.experiment_name = hyper_param_values[3]
            elif hyper_parameter_id == "training_left":
                hyper_parameter.training_steps_to_do = hyper_param_values[5]
            elif hyper_parameter_id == "learning_rate":
                hyper_parameter.learning_rate = hyper_param_values[4]
            elif hyper_parameter_id == "batch_size":
                hyper_parameter.batch_size = hyper_param_values[0]
            elif hyper_parameter_id == "eval_frequency":
                hyper_parameter.full_eval_frequency = hyper_param_values[2]
            elif hyper_parameter_id == "checkpooint_frequency":
                hyper_parameter.checkpont_frequency = hyper_param_values[1]

        request = pb2.TrainerCommand(
            hyper_parameter_change=pb2.HyperParameterCommand(
                hyper_parameters=hyper_parameter))
        stub.ExperimentCommand(request)
        return button_children

    # @app.callback(
    #     Output('layer-weights', 'children'),
    #     Input('weights-render-freq', 'n_intervals'),
    #     Input('neuron_stats-checkboxes', 'value'),
    #     # Input('refresh-weights-div-store', 'data'),
    #     State('layer-weights', 'children'),
    #     prevent_initial_call=True
    # )
    # def update_weights_div(_, checklist_values, children):
    #     print(f"[UI] WeightsLab.update_weights_div {checklist_values},")
    #     nonlocal ui_state
    #     nonlocal stub

    #     ctx = dash.callback_context
    #     if not ctx.triggered:
    #         trigger = 'No trigger'
    #     else:
    #         trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    #     print("[UI] WeightsLab.update_weights_div.trigger:", trigger)
    #     if 'weights' in trigger:
    #         print("[UI] WeightsLab.update_weights_div.refreshing.")
    #         if children:
    #             return children

    #         if not checklist_values:
    #             return []

    #     if 'neuron_stats-checkboxes' in trigger:
    #         print("[UI] WeightsLab.update_weights_div.checklist_values.")

    #         children = []
    #         for _, layer_row in ui_state.layers_df.iterrows():
    #             layer_neurons_df = ui_state.neurons_df.loc[layer_row.layer_id]
    #             layer_neurons_df = layer_neurons_df.reset_index()
    #             layer_neurons_df['layer_id'] = layer_row['layer_id']

    #             children.append(get_layer_div(
    #                 layer_row, layer_neurons_df, ui_state, checklist_values))
    #         return children
    #     return no_update

    @app.callback(
        Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'columns'),
        Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'data'),
        Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'style_data_conditional'),
        Output({'type': 'layer-div', 'layer_id': MATCH}, 'style'),
        Output({"type": "layer-sub-heading", "layer_id": MATCH}, 'children'),
        Input('weights-render-freq', 'n_intervals'),
        Input('neuron_stats-checkboxes', 'value'),
        State({'type': 'layer-data-table', 'layer_id': MATCH}, 'id'),
        State({'type': 'layer-data-table', 'layer_id': MATCH}, 'style_data_conditional'),
        State({'type': 'layer-div', 'layer_id': MATCH}, 'style'),
    )
    def update_layer_data_table(
            _, checklist_values, neuron_dt_div_id, style_data_conditional,
            layer_div_style
    ):
        # print(f"[UI] WeightsLab.update_layer_data_table.", neuron_dt_div_id)

        layer_id = neuron_dt_div_id['layer_id']
        if layer_id not in ui_state.get_neurons_df().index.get_level_values(0):
            # print('layer_id not updated:', layer_id)

            return no_update
        layer_neurons_df = ui_state.get_neurons_df().loc[layer_id].copy()
        layer_neurons_df = layer_neurons_df.reset_index()
        layer_neurons_df['layer_id'] = layer_id
        layer_row = ui_state.get_layer_df_row_by_id(layer_id)

        checklist_values = convert_checklist_to_df_head(checklist_values)
        neurons_view_df = format_values_df(layer_neurons_df[checklist_values])
        columns = [{"name": col, "id": col} for col in neurons_view_df.columns]

        highlight_conditional = []
        selected_ids = ui_state.selected_neurons[layer_row['layer_id']]

        if selected_ids:
            # Create a filter_query with multiple OR conditions
            filter_query = ' || '.join([
                f"{{neuron_id}} = '{id}'" for id in selected_ids])
            highlight_conditional = [
                {
                    "if": {"filter_query": filter_query},
                    "backgroundColor": "#ffefcc",
                }
            ]

        if len(style_data_conditional) > 6:
            style_data_conditional = style_data_conditional[-6:]
        new_cond_style = highlight_conditional + style_data_conditional

        # Handle layer_div children and style
        # We just update the second childe since it contain information about
        # the layer.
        _, sub_heading = get_layer_headings(layer_row)
        layer_width = layer_div_width(checklist_values)
        layer_div_style['minWidth'] = f"{layer_width}px"
        records = neurons_view_df.to_dict("records")

        return columns, records, new_cond_style, layer_div_style, sub_heading

    @app.callback(
        Input({"type": "layer-rem-btn", "layer_id": ALL}, "n_clicks"),
    )
    def on_layer_remove_neuron_callback(_,):
        print(f"[UI] WeightsLab.on_layer_remove_neuron_callback.")
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        layer_details = ui_state.get_layer_df_row_by_id(layer_id)
        with ui_state.lock:
            layer_row_idx = ui_state.layer_id_to_df_row_idx[layer_id]
            ui_state.layers_df.loc[layer_row_idx]['outgoing'] -= 1

        weight_operation=pb2.WeightOperation(
            op_type=pb2.WeightOperationType.REMOVE_NEURONS)

        removed_neuron_id = pb2.NeuronId(
            layer_id=layer_id,
            neuron_id=layer_details.outgoing - 1)
        weight_operation.neuron_ids.extend([removed_neuron_id])

        request = pb2.WeightsOperationRequest(weight_operation=weight_operation)
        response = stub.ManipulateWeights(request)
        print(f"Response: {response}")

    @app.callback(
        Input({"type": "layer-add-btn", "layer_id": ALL}, "n_clicks"),
    )
    def on_layer_add_neurons_callback(_,):
        print(f"[UI] WeightsLab.on_layer_add_neurons_callback.")
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        weight_operation=pb2.WeightOperation(
            op_type=pb2.WeightOperationType.ADD_NEURONS,
            layer_id=layer_id,
            neurons_to_add=1)

        request = pb2.WeightsOperationRequest(
            weight_operation=weight_operation)
        response = stub.ManipulateWeights(request)
        print(f"Response: {response}")

    @app.callback(
        Input({"type": "layer-freeze-btn", "layer_id": ALL}, "n_clicks"),
    )
    def on_layer_freeze_neuron_callback(_):
        print(f"[UI] WeightsLab.on_layer_freeze_neuron_callback")
        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        weight_operation=pb2.WeightOperation(
            op_type=pb2.WeightOperationType.FREEZE,
            layer_id=layer_id)

        request = pb2.WeightsOperationRequest(
            weight_operation=weight_operation)
        response = stub.ManipulateWeights(request)

        print(f"Freeze layer response: {response}")

    @app.callback(
        Output('modal-weights-edit', 'is_open'),
        Input({"type": "layer-see-btn", "layer_id": ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def inspect_neurons_weights_by_btn(n_clicks):
        print("[UI] Inspcting layer  weights !", n_clicks)
        if not any(click and click > 0 for click in n_clicks):
            return no_update

        ctx = dash.callback_context

        if not ctx.triggered:
            return no_update

        prop_id = ctx.triggered[0]['prop_id']
        btn_dict = eval(prop_id.split('.')[0])
        layer_id = btn_dict['layer_id']

        print("Inspecting weights of layer ", layer_id)
        return True

    @app.callback(
        # State({"type": "layer-data-table", "layer_id": MATCH}, 'data'),
        Input({"type": "layer-data-table", "layer_id": ALL}, 'selected_rows'),
        prevent_initial_call=True  # Skip initial execution
    )
    def display_weights_of_neuron(selected_rows):
        print(f"[UI] WeightsLab.display_weights_of_neuron {selected_rows}")
        if not selected_rows:
            return no_update

        selected_row_index = selected_rows
        print("selected_row_index: ", selected_row_index)
        # row = data[selected_row_index]
        # print("Selected row: ", row)

    @app.callback(
        Input('run-neuron-data-query', 'n_clicks'),
        State('neuron-query-input', 'value'),
        State('neuron-query-input-weight', 'value'),
        State('neuron-action-dropdown', "value"),
    )
    def run_query_on_neurons(_, query, weight, action):
        nonlocal ui_state
        print(
            f"[UI] WeightsLab.run_query_on_neurons {query}, {weight}, "
            f"{action} {ui_state.get_neurons_df()}")
        if weight is None:
            weight = 1.0

        selected_neurons = collections.defaultdict(lambda: [])
        try:
            selected_neurons_df = ui_state.get_neurons_df().query(query)
        except Exception as e:
            print(f"Error: {e} ", ui_state.get_neurons_df())
            return

        print("Selected neurons:", selected_neurons_df)
        sample_params = {}
        if weight <= 1.0:
            sample_params["frac"] = weight
        else:
            sample_params["n"] = weight

        selected_neurons_df = selected_neurons_df.sample(**sample_params)
        selected_neurons_df = selected_neurons_df.reset_index()

        if action == "highlight":
            for _, row in selected_neurons_df.iterrows():
                selected_neurons[row["layer_id"]].append(row["neuron_id"])
            ui_state.selected_neurons = selected_neurons
            return

        weight_operation = None
        if action == "delete":
            weight_operation=pb2.WeightOperation(
                op_type=pb2.WeightOperationType.REMOVE_NEURONS)
        elif action == "reinitialize":
            weight_operation=pb2.WeightOperation(
                op_type=pb2.WeightOperationType.REINITIALIZE)
        elif action == "freeze":
            weight_operation=pb2.WeightOperation(
                op_type=pb2.WeightOperationType.FREEZE)

        elif action == "add_neurons":
            selected_df = ui_state.get_layers_df().query(query)
            for _, row in selected_df.iterrows():
                outgoing_neurons = row['outgoing']
                layer_id = row['layer_id']

                if isinstance(weight, float) and 0 < weight < 1:
                    neurons_to_add = max(1, int(round(outgoing_neurons * weight)))
                elif isinstance(weight, int) and weight >= 1:
                    neurons_to_add = int(weight)
                else:
                    print(f"Invalid weight parameter for add_neurons: {weight}")
                    continue

                weight_operation = pb2.WeightOperation(
                    op_type=pb2.WeightOperationType.ADD_NEURONS,
                    layer_id=layer_id,
                    neurons_to_add=neurons_to_add
                )
                request = pb2.WeightsOperationRequest(weight_operation=weight_operation)
                response = stub.ManipulateWeights(request)
                print(f"Added {neurons_to_add} neurons to layer {layer_id}, response: {response}")

            return

        if weight_operation:
            for idx, row in selected_neurons_df.reset_index().iterrows():
                print("Selected neuron row: ", idx, row)
                neuron_id = pb2.NeuronId(
                    layer_id=row['layer_id'],
                    neuron_id=row['neuron_id'])
                weight_operation.neuron_ids.extend([neuron_id])

        if weight_operation:
            request = pb2.WeightsOperationRequest(
                weight_operation=weight_operation)
            print(f"Weight operation request: {request}")
            response = stub.ManipulateWeights(request)
            print(f"Weight operation response: {response}")

    @app.callback(
        Output('train-data-table', 'data'),
        Output('train-dataset-header', 'children'),
        Input('datatbl-render-freq', 'n_intervals'),
        Input('run-train-data-query', 'n_clicks'),
        State('table-refresh-checkbox', 'value'),
        State('train-data-query-input', 'value'),
    )
    def update_train_data_table(_, __, chk, query):
        if 'refresh_regularly' not in chk:
            return no_update

        df = ui_state.samples_df.copy()
        if getattr(ui_state, "task_type") == "segmentation":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "segmentation"))
        elif getattr(ui_state, "task_type") == "classification":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "classification"))

        sort_info = parse_sort_info(query)
        if sort_info:
            try:
                df = df.sort_values(by=sort_info['cols'], ascending=sort_info['dirs'])
            except Exception as e:
                print(f"[ERROR] Failed to sort train data: {e}")

        num_available_samples = (~df["Discarded"]).sum()
        return df.to_dict('records'), f"Train Dataset #{num_available_samples} samples"



    @app.callback(
        Output('eval-data-table', 'data'),
        Input('datatbl-render-freq', 'n_intervals'),
        Input('run-eval-data-query', 'n_clicks'),
        State('eval-table-refresh-checkbox', 'value'),
        State('eval-data-query-input', 'value'),
    )
    def update_eval_data_table(_, __, chk, query):
        if 'refresh_regularly' not in chk:
            return no_update

        df = ui_state.eval_samples_df.copy()
        if getattr(ui_state, "task_type", "classification") == "segmentation":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "segmentation"))
        elif getattr(ui_state, "task_type", "classification") == "classification":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: format_for_table(v, "classification"))

        sort_info = parse_sort_info(query)
        if sort_info:
            try:
                df = df.sort_values(by=sort_info['cols'], ascending=sort_info['dirs'])
            except Exception as e:
                print(f"[ERROR] Failed to sort eval data: {e}")
        return df.to_dict('records')

    @app.callback(
        Input('run-train-data-query', 'n_clicks'),
        Input('run-eval-data-query', 'n_clicks'),
        State('train-data-query-input', 'value'),
        State('eval-data-query-input', 'value'),
        State('data-query-input-weight', 'value'),
        State('eval-data-query-weight', 'value'),
        State('train-query-discard-toggle', 'value'),
        State('eval-query-discard-toggle', 'value'),
        State('train-denylist-accumulate-checkbox', 'value'),
        State('eval-denylist-accumulate-checkbox', 'value'),
        prevent_initial_call=True
    )
    def run_query_on_dataset(train_click, eval_click,
                            train_query, eval_query,
                            train_weight, eval_weight,
                            train_toggle, eval_toggle,
                            train_accumulate, eval_accumulate):
        ctx_triggered = dash.callback_context.triggered
        if not ctx_triggered:
            return no_update

        triggered_id = ctx_triggered[0]['prop_id'].split('.')[0]

        if 'run-train-data-query' in triggered_id:
            tab_type = "train"
            query = train_query
            weight = train_weight
            toggle_values = train_toggle
        else:
            tab_type = "eval"
            query = eval_query
            weight = eval_weight
            toggle_values = eval_toggle

        if not query or 'sortby' in query.lower():
            return no_update
        if weight is None:
            weight = 1.0
        un_discard = 'undiscard' in toggle_values

        try:
            df, remove_op_key, deny_op_key = get_query_context(tab_type, ui_state)
            task_type = getattr(ui_state, "task_type", "classification")
            if task_type == "classification":
                for col in ["Prediction", "Target"]:
                    if col in df.columns:
                        df[col] = df[col].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v)
            elif task_type == "segmentation":
                for col in ["Prediction", "Target"]:
                    if col in df.columns:
                        df[col + "ClassesStr"] = df[col].apply(lambda arr: ",".join([str(x) for x in arr]))
            filter_fn = rewrite_query_for_lists(query, task_type, ui_state)
            if filter_fn:
                query_dataframe = filter_fn(df)
            else:
                query_dataframe = df.query(query)

            if weight <= 1.0:
                query_dataframe = query_dataframe.sample(frac=weight)
            elif isinstance(weight, int):
                query_dataframe = query_dataframe.sample(n=weight)

            sample_ids = query_dataframe['SampleId'].to_list()
            deny_op = pb2.DenySamplesOperation()
            deny_op.sample_ids.extend(sample_ids)
            request = pb2.TrainerCommand()
            accumulate = 'accumulate' in (train_accumulate if tab_type == "train" else eval_accumulate)
            if un_discard:
                if tab_type == "train":
                    request.remove_from_denylist_operation.CopyFrom(deny_op)
                else:
                    request.remove_eval_from_denylist_operation.CopyFrom(deny_op)
            else:
                if tab_type == "train":
                    request.deny_samples_operation.CopyFrom(deny_op)
                    request.deny_samples_operation.accumulate = accumulate
                else:
                    request.deny_eval_samples_operation.CopyFrom(deny_op)
                    request.deny_eval_samples_operation.accumulate = accumulate

            response = stub.ExperimentCommand(request)
            print(
                f"[{tab_type.capitalize()} Query] {query}, Weight: {weight}, Un-discard: {un_discard}, "
                f"Sample count: {len(sample_ids)}, Response: {response.message}"
            )

        except Exception as e:
            print(f"[ERROR] {tab_type.capitalize()} query failed: {e}")

        return no_update


    @app.callback(
        Input('run-eval-data-query', 'n_clicks'),
        State('eval-data-query-input', 'value'),
        State('eval-data-query-weight', 'value'),
        State('eval-query-discard-toggle', 'value'),
        prevent_initial_call=True
    )
    def run_eval_query_on_dataset(_, query, weight, toggle_values):
        if 'sortby' in query.lower():
            return no_update
        
        if weight is None:
            weight = 1.0

        un_discard = 'undiscard' in toggle_values
        try:
            query_dataframe = ui_state.eval_samples_df.query(query)

            if weight <= 1.0:
                query_dataframe = query_dataframe.sample(frac=weight)
            elif isinstance(weight, int):
                query_dataframe = query_dataframe.sample(n=weight)

            sample_ids = query_dataframe['SampleId'].to_list()

            if un_discard:
                allow_op = pb2.DenySamplesOperation()
                allow_op.sample_ids.extend(sample_ids)
                request = pb2.TrainerCommand(
                    remove_eval_from_denylist_operation=allow_op
                )
            else:
                deny_op = pb2.DenySamplesOperation()
                deny_op.sample_ids.extend(sample_ids)
                request = pb2.TrainerCommand(
                    deny_eval_samples_operation=deny_op
                )

            response = stub.ExperimentCommand(request)
            print(
                f"[Eval Query] {query}, Weight: {weight}, Un-discard: {un_discard}, "
                f"Sample count: {len(sample_ids)}, Response: {response.message}"
            )

        except Exception as e:
            print(f"[ERROR] Eval query failed: {e}")

        return no_update


    @app.callback(
        Output('train-data-div', 'style', allow_duplicate=True),
        Input('sample-inspect-checkboxes', 'value'),
        State('train-data-div', 'style'),
    )
    def update_train_data_div_style(inspect_checkboxes, old_div_style):
        print(f"[UI] WeightsLab.update_train_data_div_style {inspect_checkboxes}")
        width_percent = 45
        width_percent_delta = (90 - width_percent) // 2

        total_new_width_percent = width_percent + \
            len(inspect_checkboxes) * width_percent_delta
        style = dict(old_div_style)

        style.update({
            'width': f'{total_new_width_percent}vw',
            'maxWdith': f'{total_new_width_percent+2}vw',
        })

        return style

    # @app.callback(
    #     Output('data-panel-col1', 'children', allow_duplicate=True),
    #     Input('train-data-table', 'selected_rows'),
    #     State('train-data-table', 'data'),
    #     State('sample-inspect-checkboxes', 'value'),
    # )
    # def render_data_sample(selected_rows, data, inspect_checkboxes):
    #     if selected_rows is None or len(selected_rows) == 0 or \
    #             len(inspect_checkboxes) == 0:
    #         return []

    #     # Get the selected row's data
    #     selected_row_index = selected_rows[-1]
    #     row = data[selected_row_index]
    #     selected_sample_id = row["SampleId"]
    #     request = pb2.SampleRequest(
    #         sample_id=selected_sample_id, origin="train")
    #     response = stub.GetSample(request)

    #     image_base64 = base64.b64encode(response.data).decode('utf-8')

    #     return html.Img(
    #         src=f'data:image/png;base64,{image_base64}',
    #         style={
    #             'width': '18vw',
    #             'height': '18vh',
    #             'marginTop': '10vh',
    #         }
    #     )

    @app.callback(
        Output('train-data-table', 'page_size'),
        Input('grid-preset-dropdown', 'value')
    )
    def update_page_size(grid_count):
        return grid_count

    @app.callback(
        Output('eval-data-table', 'page_size'),
        Input('eval-grid-preset-dropdown', 'value')
    )
    def update_eval_page_size(grid_count):
        return grid_count


    @app.callback(
        Output('train-sample-panel', 'children', allow_duplicate=True),
        Output('eval-sample-panel', 'children', allow_duplicate=True),
        Input('train-data-table', 'derived_viewport_data'),
        Input('train-data-table', 'selected_rows'),
        Input('eval-data-table', 'derived_viewport_data'),
        Input('eval-data-table', 'selected_rows'),
        Input('sample-inspect-checkboxes', 'value'),
        Input('eval-sample-inspect-checkboxes', 'value'),
        Input('data-tabs', 'value'),
        prevent_initial_call=True
    )
    def render_samples(
        train_viewport, train_selected_rows,
        eval_viewport, eval_selected_rows,
        train_flags, eval_flags,
        tab
    ):
        panels = [no_update, no_update]
        nonlocal ui_state, stub
        if tab == 'train' and 'inspect_sample_on_click' in train_flags and train_viewport:
            df = ui_state.samples_df
            ids = [row['SampleId'] for row in train_viewport if row['SampleId'] in df['SampleId'].values]
            selected_ids = set(df.iloc[i]['SampleId'] for i in train_selected_rows or [])
            discarded_ids = set(df.loc[df['Discarded'], 'SampleId'])
            panels[0] = render_images(ui_state, stub, ids, selected_ids, origin='train', discarded_ids=discarded_ids)
        elif tab == 'eval' and 'inspect_sample_on_click' in eval_flags and eval_viewport:
            df = ui_state.eval_samples_df
            ids = [row['SampleId'] for row in eval_viewport if row['SampleId'] in df['SampleId'].values]
            selected_ids = set(df.iloc[i]['SampleId'] for i in eval_selected_rows or [])
            discarded_ids = set(df.loc[df['Discarded'], 'SampleId'])
            panels[1] = render_images(ui_state, stub, ids, selected_ids, origin='eval', discarded_ids=discarded_ids)
        return panels


    @app.callback(
        Output('train-data-table', 'data', allow_duplicate=True),
        Input('train-data-table', 'data'),
        State('train-data-table', 'data_previous'),
        State('table-refresh-checkbox', 'value')
    )
    def denylist_deleted_rows_sample_ids(
        current_data, previous_data, table_checkboxes):
        if previous_data is None or len(previous_data) == 0:
            return no_update

        previous_sample_ids = set([row["SampleId"] for row in previous_data])
        current_sample_ids = set([row["SampleId"] for row in current_data])

        diff_sample_ids = previous_sample_ids - current_sample_ids

        for row in previous_data:
            if row["SampleId"] in diff_sample_ids:
                row['Discarded'] = True

        sample_deny_request = pb2.TrainerCommand(
            deny_samples_operation=pb2.DenySamplesOperation(
                sample_ids=list(diff_sample_ids))
        )
        sample_deny_response = stub.ExperimentCommand(sample_deny_request)
        del sample_deny_response

        if "discard_by_flag_flip" in table_checkboxes:
            return previous_data
        return current_data
    
    @app.callback(
        Output('train-image-selected-ids', 'data'),
        Input('train-data-table', 'selected_rows'),
        State('train-data-table', 'data'),
        prevent_initial_call=True
    )
    def store_selected_sample_ids(selected_rows, data):
        if not selected_rows or not data:
            return []
        return [data[i]['SampleId'] for i in selected_rows]

    @app.callback(
        Output('train-data-table', 'selected_rows', allow_duplicate=True),
        Input('train-image-selected-ids', 'data'),
        Input('train-data-table', 'data'),
        prevent_initial_call=True
    )
    def restore_selected_rows(selected_ids, data):
        if not selected_ids or not data:
            return []
        id2idx = {row['SampleId']: idx for idx, row in enumerate(data)}
        return [id2idx[sid] for sid in selected_ids if sid in id2idx]

    @app.callback(
        Output('train-data-table', 'style_data_conditional'),
        Input('train-data-table', 'selected_rows'),
        State('train-data-table', 'data')
    )
    def highlight_selected(selected_rows, data):
        if not selected_rows or not data:
            return []
        sample_ids = [data[i]['SampleId'] for i in selected_rows]
        filter_query = ' || '.join([f'{{SampleId}} = {sid}' for sid in sample_ids])
        return [{
            "if": {"filter_query": filter_query},
            "backgroundColor": "#ffe6b3",
            "fontWeight": "bold"
        }] if sample_ids else []

    @app.callback(
        Output('experiment_checklist', 'options', allow_duplicate=True),
        Output('experiment_checklist', 'value', allow_duplicate=True),
        Input('graphss-render-freq', 'n_intervals'),
    )
    def update_experiments_checklist(n_intervals):
        nonlocal ui_state

        experiment_names = list(ui_state.exp_names)
        options = [
            {'label': experiment_name, 'value': experiment_name}
            for experiment_name in experiment_names]
        return options, experiment_names

    @app.callback(
        Output("experiment_plots_div", "children"),
        Input("graphss-render-freq", "n_intervals"),
        State("experiment_plots_div", "children")
    )
    def add_graphs_to_div(_, existing_children):
        # print(f"UI.add_graphs_to_div")
        nonlocal ui_state

        graph_names = sorted(ui_state.met_names)

        if len(graph_names) == len(existing_children):
            return existing_children
        if len(graph_names) == 0:
            return no_update

        graph_divs = []
        for graph_name in graph_names:
            graph_divs.append(
                dcc.Graph(
                    id={"type": "graph", "index": graph_name},
                    config={"displayModeBar": False},
                )
            )
        return graph_divs

    @app.callback(
        Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
        Input("graphss-render-freq", "n_intervals"),
        State({'type': "graph", "index": MATCH}, "id"),
        State('experiment_checklist', "value"),
        # State("plot-smoothness-slider", "value"),
        prevent_initial_call=True,
    )
    def update_graph(_, graph_id, checklist):
        # print("update_graph", graph_id, checklist)
        nonlocal ui_state

        metric_name = graph_id["index"]
        data = []

        for experiment_name in checklist:
            data.extend(ui_state.get_plots_for_exp_name_metric_name(
                metric_name, experiment_name)
            )
        if ui_state.plot_name_2_curr_head_point[metric_name] is not None:
            curr_point = ui_state.plot_name_2_curr_head_point[metric_name]
            data.append(
                go.Scattergl(
                    x=[curr_point.x],
                    y=[curr_point.y],
                    mode='markers',
                    name="Current Model",
                    marker_symbol="star-diamond-open-dot",
                    marker=dict(color='red', size=16)
                )
            )

        select_graph = go.Scattergl(
            x=[None],
            y=[None],
            mode='markers',
            name="",
            marker_symbol="diamond",
            marker=dict(color='cyan', size=16, opacity=0.8)
        )

        figure = {
            'data': data + [select_graph],
            'layout': go.Layout(
                title=metric_name,
                xaxis={'title': 'Seen Samples'},
                yaxis={'title': "Value"},
            )
        }
        return figure

    @app.callback(
        Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
        Output({'type': "graph", "index": MATCH}, "ClickData"),
        [
            Input({'type': "graph", "index": MATCH}, 'hoverData'),
            Input({'type': "graph", "index": MATCH}, 'clickData'),
        ],
        State({'type': "graph", "index": MATCH}, "figure"),
        prevent_initial_call = True
    )
    def update_selection_of_checkpoint(hoverData, clickData, figure):
        nonlocal stub
        nonlocal ui_state
        # print("update_selection_of_checkpoint", hoverData, clickData, figure)

        ctx = dash.callback_context
        if not ctx.triggered:
            return figure, no_update

        if hoverData is None or 'points' not in hoverData:
            return no_update

        cursor_x = hoverData['points'][0]['x']
        cursor_y = hoverData['points'][0]['y']

        x_min, y_min, t_min, i_min, min_dist = None, None, None, None, 1e10

        if 'data' not in figure:
            return no_update

        for t_idx, trace_data in enumerate(figure['data']):
            if "ckpt" not in trace_data['name']:
                continue
            x_data = np.array(trace_data['x'])
            y_data = np.array(trace_data['y'])

            for i, val in enumerate(x_data):
                x_data[i] = 0 if val is None else val

            for i, val in enumerate(y_data):
                y_data[i] = 0 if val is None else val

            if len(y_data) < len(x_data):
                x_data = x_data[:-1]
            elif len(x_data) < len(y_data):
                y_data = y_data[:-1]

            if x_data is None or y_data is None or x_data.size == 0 or \
                    y_data.size == 0 or cursor_x is None or cursor_y is None:
                continue

            # replace None in x_data and y_data with 0
            x_data = np.nan_to_num(x_data)
            try:
                distances = np.sqrt(
                    (x_data - cursor_x) ** 2 + (y_data - cursor_y) ** 2)
                min_index = np.argmin(distances)  # Index of the closest point
                if distances[min_index] < min_dist:
                    x_min, y_min, t_min, i_min, min_dist = (
                        x_data[min_index], y_data[min_index], t_idx, min_index,
                        distances[min_index])
            except Exception as e:
                print(f"Error in update_selection_of_checkpoint: {e}")
                continue

        checkpoint_id_to_load = None
        if t_min is not None and i_min is not None:
            figure['data'][-1]['x'] = [x_min]
            figure['data'][-1]['y'] = [y_min]

            if i_min < len(figure['data'][t_min]["customdata"]):
                checkpoint_id_to_load = \
                    figure['data'][t_min]["customdata"][i_min]

        trigger = ctx.triggered[0]["prop_id"]
        if "clickData" in trigger and clickData:
            load_checkpoint_op = pb2.LoadCheckpointOperation(
                checkpoint_id=checkpoint_id_to_load)
            load_checkpoint_request = pb2.TrainerCommand(
                load_checkpoint_operation=load_checkpoint_op)
            ckpt_load_result = stub.ExperimentCommand(
                load_checkpoint_request)

            print(f"Checkpoint load result: {ckpt_load_result}")
            ui_state.current_run_id += 1

            update_request = pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True
            )
            updated_state = stub.ExperimentCommand(update_request)
            ui_state.update_from_server_state(updated_state)

            if checkpoint_id_to_load is not None:
                print("Figure data: ", figure['data'][t_min])

            return figure, None

        return figure, no_update


    app.run(debug=False, port=8050)

if __name__ == '__main__':
    main()