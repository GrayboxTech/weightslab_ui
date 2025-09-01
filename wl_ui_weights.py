import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, ctx, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import threading
import grpc
import time
import argparse
import collections
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import (
    UIState, get_weights_div,
    convert_checklist_to_df_head, format_values_df,
    get_layer_headings, layer_div_width,
    get_pause_play_button,
    get_header_hyper_params_div,

)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_directory",
        type=str,
        required=True,
        help="Path to the experiment root directory",
    )
    return parser.parse_args()

args = parse_args()
channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.ExperimentServiceStub(channel)
ui_state = UIState(args.root_directory)

initial_state = stub.ExperimentCommand(pb2.TrainerCommand(
    get_hyper_parameters=True,
    get_interactive_layers=True,
))
ui_state.update_from_server_state(initial_state)

def pause_training():
    hyper_parameter = pb2.HyperParameters()
    hyper_parameter.is_training = False
    request = pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(
            hyper_parameters=hyper_parameter))
    stub.ExperimentCommand(request)
    time.sleep(0.3)

def refresh_ui_state():
    while True:
        try:
            update_request = pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True,
            )
            updated_state = stub.ExperimentCommand(update_request)
            ui_state.update_from_server_state(updated_state)
        except Exception as e:
            print("Error updating UI state:", e)

thread = threading.Thread(target=refresh_ui_state, daemon=True)
thread.start()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "WeightsLab - Architecture Only"

app.layout = html.Div([
    dcc.Interval(id='weights-render-freq', interval=5000, n_intervals=0),
    html.H1("Model Architecture", style={'textAlign': 'center'}),
    get_header_hyper_params_div(ui_state),
    get_weights_div(ui_state)
])

def _get_next_layer_id(curr_layer_id: int) -> int | None:
    layers_df = ui_state.get_layers_df().sort_values("layer_id")
    ids = list(layers_df["layer_id"].values)
    if curr_layer_id not in ids:
        return None
    idx = ids.index(curr_layer_id)
    return ids[idx + 1] if idx + 1 < len(ids) else None

def _get_incoming_count(layer_id: int) -> int | None:
    row = ui_state.get_layer_df_row_by_id(layer_id)
    if isinstance(row, pd.Series) and 'incoming' in row.index:
        val = int(row['incoming'])
        return val
    return None

def _make_heatmap_figure(z, zmin=None, zmax=None):
    return go.Figure(
        data=[go.Heatmap(
            z=z, zmid=0, zmin=zmin, zmax=zmax,
            colorscale=[[0.0, 'red'], [0.5, 'white'], [1.0, 'green']],
            showscale=False
        )]
    ).update_layout(
        margin=dict(l=2, r=2, t=2, b=2),
        xaxis_showgrid=False, yaxis_showgrid=False,
        xaxis_visible=False, yaxis_visible=False,
    )

def _downsample_strip(z_1xN: np.ndarray, max_len: int = 256) -> np.ndarray:
    N = z_1xN.shape[1]
    if N <= max_len:
        return z_1xN
    bucket = int(np.ceil(N / max_len))
    trim = (N // bucket) * bucket
    if trim == 0:
        return z_1xN
    v = z_1xN[0, :trim].reshape(-1, bucket).mean(axis=1)
    return v.reshape(1, -1)

def _parse_chw(s):
        if not s or not isinstance(s, str):
            return None
        import re
        m = re.match(r'^\s*(\d+)\s*[x×]\s*(\d+)\s*[x×]\s*(\d+)\s*$', s, re.I)
        if not m:
            return None
        C, H, W = map(int, m.groups())
        return (C, H, W)


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
            button_children = get_pause_play_button()
            hyper_parameter.training_steps_to_do = hyper_param_values[5]
        else:
            button_children = get_pause_play_button()
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

@app.callback(
    Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'columns'),
    Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'data'),
    Output({'type': 'layer-data-table', 'layer_id': MATCH}, 'style_data_conditional'),
    Output({'type': 'layer-div', 'layer_id': MATCH}, 'style'),
    Output({'type': 'layer-sub-heading', 'layer_id': MATCH}, 'children'),
    Input('weights-render-freq', 'n_intervals'),
    Input('neuron_stats-checkboxes', 'value'),
    State({'type': 'layer-data-table', 'layer_id': MATCH}, 'id'),
    State({'type': 'layer-data-table', 'layer_id': MATCH}, 'style_data_conditional'),
    State({'type': 'layer-div', 'layer_id': MATCH}, 'style'),
)
def update_layer_data_table(_, checklist_values, neuron_dt_div_id, style_data_conditional, layer_div_style):
    layer_id = neuron_dt_div_id['layer_id']
    if layer_id not in ui_state.get_neurons_df().index.get_level_values(0):
        return dash.no_update

    layer_neurons_df = ui_state.get_neurons_df().loc[layer_id].copy().reset_index()
    layer_neurons_df['layer_id'] = layer_id
    layer_row = ui_state.get_layer_df_row_by_id(layer_id)

    checklist_values = convert_checklist_to_df_head(checklist_values)
    neurons_view_df = format_values_df(layer_neurons_df[checklist_values])
    columns = [{"name": col, "id": col} for col in neurons_view_df.columns]
    records = neurons_view_df.to_dict("records")

    highlight_conditional = []
    selected_ids = ui_state.selected_neurons[layer_row['layer_id']]

    if selected_ids:
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

    _, sub_heading = get_layer_headings(layer_row)
    layer_width = layer_div_width(checklist_values)
    layer_div_style['minWidth'] = f"{layer_width}px"

    return columns, records, new_cond_style, layer_div_style, sub_heading


@app.callback(
    Output({'type': 'layer-add-btn', 'layer_id': ALL}, 'n_clicks'),
    Input({'type': 'layer-add-btn', 'layer_id': ALL}, 'n_clicks'),
    State('zerofy-options-checklist', 'value')
)
def add_neuron(n_clicks, zerofy_opts):
    if not ctx.triggered:
        return dash.no_update
    triggered = ctx.triggered_id
    if not triggered:
        return dash.no_update

    layer_id = triggered['layer_id']  
    n_add = 1

    next_layer_id = _get_next_layer_id(layer_id)
    old_incoming = _get_incoming_count(next_layer_id) if next_layer_id is not None else None

    add_op = pb2.WeightOperation(
        op_type=pb2.WeightOperationType.ADD_NEURONS,
        layer_id=layer_id,
        neurons_to_add=n_add
    )
    stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=add_op))

    if next_layer_id is None or old_incoming is None:
        return n_clicks

    new_from_ids = list(range(old_incoming, old_incoming + n_add))

    selected_to_ids = ui_state.selected_neurons[next_layer_id] or []
    predicates = []
    if zerofy_opts:
        if 'frozen' in zerofy_opts:
            predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_FROZEN)
        if 'older' in zerofy_opts:
            predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_OLDER)


    if not selected_to_ids and not predicates:
        print("[UI] No ZEROFY targets (no selection & no predicates).")
        return n_clicks

    zerofy_op = pb2.WeightOperation(
        op_type=pb2.WeightOperationType.ZEROFY,
        layer_id=next_layer_id,
        zerofy_from_incoming_ids=new_from_ids,
        zerofy_to_neuron_ids=selected_to_ids
    )
    if predicates:
        zerofy_op.zerofy_predicates.extend(predicates)

    resp = stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=zerofy_op))
    print(resp.message)

    ui_state.update_from_server_state(
        stub.ExperimentCommand(pb2.TrainerCommand(
            get_hyper_parameters=True,
            get_interactive_layers=True,
        ))
    )
    return n_clicks

@app.callback(
    Output({'type': 'layer-rem-btn', 'layer_id': ALL}, 'n_clicks'),
    Input({'type': 'layer-rem-btn', 'layer_id': ALL}, 'n_clicks')
)
def remove_neuron(n_clicks):
    if not ctx.triggered: return dash.no_update
    triggered = ctx.triggered_id
    if not triggered: return dash.no_update

    layer_id = triggered['layer_id']
    layer_row = ui_state.get_layer_df_row_by_id(layer_id)
    op = pb2.WeightOperation(
        op_type=pb2.WeightOperationType.REMOVE_NEURONS,
        neuron_ids=[pb2.NeuronId(layer_id=layer_id, neuron_id=layer_row.outgoing - 1)]
    )
    stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=op))
    return n_clicks
@app.callback(
    Input('run-neuron-data-query', 'n_clicks'),
    State('neuron-query-input', 'value'),
    State('neuron-query-input-weight', 'value'),
    State('neuron-action-dropdown', "value"),
    State('zerofy-options-checklist', 'value')
    )
def run_query_on_neurons(_, query, weight, action, zerofy_opts):
    print(f"[UI] WeightsLab.run_query_on_neurons {query}, {weight}, {action}")
    global ui_state
    if weight is None:
        weight = 1.0

    selected_neurons = collections.defaultdict(lambda: [])
    try:
        selected_neurons_df = ui_state.neurons_df.query(query)
    except Exception as e:
        print(f"Error: {e} ", ui_state.neurons_df)
        return

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
            layer_id = int(row['layer_id'])
            outgoing_neurons = int(row['outgoing'])

            if isinstance(weight, float) and 0 < weight < 1:
                neurons_to_add = max(1, int(round(outgoing_neurons * weight)))
            elif isinstance(weight, int) and weight >= 1:
                neurons_to_add = int(weight)
            else:
                print(f"[UI][query add] Invalid weight for add_neurons: {weight}")
                continue

            next_layer_id = _get_next_layer_id(layer_id)
            old_incoming = _get_incoming_count(next_layer_id) if next_layer_id is not None else None

            add_op = pb2.WeightOperation(
                op_type=pb2.WeightOperationType.ADD_NEURONS,
                layer_id=layer_id,
                neurons_to_add=neurons_to_add
            )
            stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=add_op))

            if next_layer_id is None or old_incoming is None:
                continue

            new_from_ids = list(range(old_incoming, old_incoming + neurons_to_add))

            selected_to_ids = ui_state.selected_neurons[next_layer_id] or []
            predicates = []
            if zerofy_opts:
                if 'frozen' in zerofy_opts:
                    predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_FROZEN)
                if 'older' in zerofy_opts:
                    predicates.append(pb2.ZerofyPredicate.ZEROFY_PREDICATE_WITH_OLDER)

            if not selected_to_ids and not predicates:
                print("[UI][query add] No ZEROFY targets (no selection & no predicates).")
                continue

            zerofy_op = pb2.WeightOperation(
                op_type=pb2.WeightOperationType.ZEROFY,
                layer_id=next_layer_id,
                zerofy_from_incoming_ids=new_from_ids,
                zerofy_to_neuron_ids=selected_to_ids
            )
            if predicates:
                zerofy_op.zerofy_predicates.extend(predicates)

            resp = stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=zerofy_op))
            print(resp.message)

        ui_state.update_from_server_state(
            stub.ExperimentCommand(pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True,
            ))
        )
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
        # UI state refresh immediately
        ui_state.update_from_server_state(
            stub.ExperimentCommand(pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True,
            ))
        )

@app.callback(
    Output({'type': 'layer-side-panel', 'layer_id': MATCH}, 'style'),
    Input('neuron_stats-checkboxes', 'value'),
    State({'type': 'layer-side-panel', 'layer_id': MATCH}, 'style'),
)
def toggle_side_panel(checklist_values, style):
    values = checklist_values or []
    show = ('show_activation_maps' in values) or ('show_filter_heatmaps' in values) or ('show_heatmaps' in values)
    style = dict(style or {})
    style['display'] = 'block' if show else 'none'
    style['maxHeight'] = '300px'
    style['overflowY'] = 'auto'
    return style


@app.callback(
    Output({'type': 'layer-activation', 'layer_id': MATCH}, 'children'),
    Output({'type': 'layer-activation', 'layer_id': MATCH}, 'style'),
    Input('weights-render-freq', 'n_intervals'),
    Input('neuron_stats-checkboxes', 'value'),
    State({'type': 'layer-activation', 'layer_id': MATCH}, 'id'),
)
def render_layer_activation(_, checklist_values, act_id):
    values = checklist_values or []
    if 'show_activation_maps' not in values:
        return dash.no_update, {'display': 'none'}

    layer_id = int(act_id['layer_id'])
    sample_id, origin = 0, "eval"

    resp = stub.GetActivations(pb2.ActivationRequest(
        layer_id=layer_id, sample_id=sample_id, origin=origin, pre_activation=True
    ))

    count = int(getattr(resp, "neurons_count", 0) or 0)
    if count <= 0:
        block = html.Div([html.Small("No activations available")],
                        style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
        return [block], {'display': 'block'}

    graphs = []
    if "Conv2d" in (resp.layer_type or ""):
        for i in range(count):
            amap = resp.activations[i]
            vals = np.array(amap.values, dtype=float).reshape(amap.H, amap.W)
            max_abs = float(np.max(np.abs(vals))) if vals.size else 1.0
            fig = _make_heatmap_figure(vals, zmin=-max_abs, zmax=+max_abs)
            graphs.append(
                html.Div(
                    dcc.Graph(figure=fig, config={'displayModeBar': False},
                            style={'height': '40px', 'width': '40px'}),
                    style={'display': 'inline-block'}
                )
            )
    else:
        scalars = np.array([resp.activations[i].values[0] for i in range(count)], dtype=float)
        max_abs = float(np.max(np.abs(scalars))) if scalars.size else 1.0
        z = scalars.reshape(1, -1)  # (1, N)
        fig = _make_heatmap_figure(z, zmin=-max_abs, zmax=+max_abs)
        width = max(20, min(20 * z.shape[1], 600))
        graphs.append(
            html.Div(
                dcc.Graph(figure=fig, config={'displayModeBar': False},
                        style={'height': '20px', 'width': f'{width}px'}),
                style={'display': 'inline-block'}
            )
        )

    block = html.Div(
        [
            html.Div(
                graphs,
                style={
                    'display': 'grid',
                    'gap': '4px',
                    'maxHeight': '420px',
                    'overflowY': 'auto',
                    'paddingRight': '6px'
                }
            )
        ],
        style={'borderTop': '1px solid #eee', 'paddingTop': '6px'}
    )
    return [block], {'display': 'block'}



@app.callback(
    Output('activation-sample-id', 'max'),
    Output('activation-sample-count', 'children'),
    Input('activation-origin', 'value'),
)
def update_sample_bounds(origin):
    try:
        resp = stub.ExperimentCommand(pb2.TrainerCommand(get_data_records=origin))
        n = int(resp.sample_statistics.sample_count or 0)
        max_id = max(n - 1, 0)
        return max_id, f"ID range: 0–{max_id} ({origin})"
    except Exception as e:
        return no_update, f"(couldn’t fetch sample count: {e})"

@app.callback(
    Output({'type': 'layer-heatmap', 'layer_id': MATCH}, 'children'),
    Output({'type': 'layer-heatmap', 'layer_id': MATCH}, 'style'),
    Input('weights-render-freq', 'n_intervals'),
    Input('neuron_stats-checkboxes', 'value'),
    State({'type': 'layer-heatmap', 'layer_id': MATCH}, 'id'),
    Input('linear-incoming-shape', 'value'),
)

def render_layer_heatmap(_, checklist_values, heatmap_id, linear_shape_text):
    values = checklist_values or []
    if ('show_filter_heatmaps' not in values) and ('show_heatmaps' not in values):
        return dash.no_update, {'display': 'none'}

    layer_id = int(heatmap_id['layer_id'])

    resp = stub.GetWeights(pb2.WeigthsRequest(
        neuron_id=pb2.NeuronId(layer_id=layer_id, neuron_id=-1)
    ))

    if not resp.success:
        msg = getattr(resp, "error_message", "Unknown error")
        block = html.Div([html.Small(msg)],
                        style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
        return [block], {'display': 'block'}

    layer_type = (resp.layer_type or "").strip()
    C_in, C_out = int(resp.incoming), int(resp.outgoing)
    w = np.array(resp.weights, dtype=float)

    tiles_by_neuron = []

    if "Conv2d" in layer_type:
        K = int(resp.kernel_size or 0)
        expected = C_out * C_in * K * K
        if K <= 0 or w.size != expected:
            msg = (f"Unexpected weight shape: got {w.size}, expected {expected} "
                f"(C_out={C_out}, C_in={C_in}, K={K})")
            block = html.Div([html.Small(msg)],
                            style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
            return [block], {'display': 'block'}
        w = w.reshape(C_out, C_in, K, K)
        for out_id in range(C_out):
            tiles_by_neuron.append([w[out_id, in_id] for in_id in range(C_in)])

    else:
        expected = C_out * C_in
        if w.size != expected:
            msg = (f"Unexpected weight shape: got {w.size}, expected {expected} "
                f"(C_out={C_out}, C_in={C_in})")
            block = html.Div([html.Small(msg)],
                            style={'borderTop': '1px solid #eee', 'paddingTop': '6px'})
            return [block], {'display': 'block'}

        w = w.reshape(C_out, C_in)  # (out, in)
        CHW = _parse_chw(linear_shape_text)
        can_reshape = CHW is not None and (CHW[0] * CHW[1] * CHW[2] == C_in)

        if can_reshape:
            C, H, W = CHW
            for out_id in range(C_out):
                vol = w[out_id, :].reshape(C, H, W)      # (C,H,W)
                tiles_by_neuron.append([vol[c] for c in range(C)])  # one H×W per input channel
        else:
            for out_id in range(C_out):
                strip = _downsample_strip(w[out_id, :].reshape(1, C_in), max_len=256)
                tiles_by_neuron.append([strip])

    rows = []
    for row_tiles in tiles_by_neuron:
        row_graphs = []
        for z in row_tiles:
            max_abs = float(np.max(np.abs(z))) if z.size else 1.0
            fig = _make_heatmap_figure(z, zmin=-max_abs, zmax=+max_abs)
            if z.ndim == 2 and z.shape[0] == 1:  # strip
                width = max(40, min(10 * z.shape[1], 600))
                style = {'height': '16px', 'width': f'{width}px'}
            else:                                 # square map (K×K or H×W)
                style = {'height': '36px', 'width': '36px'}
            row_graphs.append(
                html.Div(dcc.Graph(figure=fig, config={'displayModeBar': False}, style=style),
                        style={'display': 'inline-block', 'marginRight': '4px'})
            )
        rows.append(html.Div(row_graphs, style={'whiteSpace': 'nowrap', 'overflowX': 'auto', 'marginBottom': '6px'}))

    block = html.Div(rows, style={
        'borderTop': '1px solid #eee', 'paddingTop': '6px',
        'maxHeight': '420px', 'overflowY': 'auto', 'paddingRight': '6px'
    })
    return [block], {'display': 'block'}




if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
