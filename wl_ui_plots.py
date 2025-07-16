import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, ctx, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import threading
import grpc
import time
import argparse
import numpy as np
import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
import plotly.graph_objs as go

from weights_lab import (
    UIState,
    get_plots_div,
    get_header_hyper_params_div,
    get_pause_play_button
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True)
    return parser.parse_args()

args = parse_args()
channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.ExperimentServiceStub(channel)
ui_state = UIState(args.root_directory)

# Initial fetch
initial_state = stub.ExperimentCommand(
    pb2.TrainerCommand(
        get_hyper_parameters=True,
        get_interactive_layers=True,
        # get_data_records="train"
    )
)
ui_state.update_from_server_state(initial_state)

def refresh_ui_state():
    while True:
        try:
            update_request = pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers = True)
            updated_state = stub.ExperimentCommand(update_request)
            ui_state.update_from_server_state(updated_state)
        except Exception as e:
            print("Error updating UI state:", e)
        time.sleep(5)

threading.Thread(target=refresh_ui_state, daemon=True).start()

def retrieve_training_statuses():
    global ui_state, stub
    for status in stub.StreamStatus(pb2.Empty()):
        ui_state.update_metrics_from_server(status)

threading.Thread(target=retrieve_training_statuses, daemon=True).start()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "WeightsLab - Plots Only"

app.layout = html.Div([
    dcc.Interval(id='graphss-render-freq', interval=10000, n_intervals=0),
    html.H1("Experiment Plots", style={'textAlign': 'center'}),
    get_header_hyper_params_div(ui_state),
    get_plots_div()
])

@app.callback(
    Output('resume-pause-train-btn', 'children', allow_duplicate=True),
    Input({"type": "hyper-params-input", "idx": ALL}, "value"),
    Input('resume-pause-train-btn', 'n_clicks'),
    prevent_initial_call=True
)
def send_to_controller_hyper_parameters_on_change(hyper_param_values, resume_pause_clicks):
    print(f"[UI] PlotsOnly.send_to_controller_hyper_parameters_on_change {hyper_param_values}, {resume_pause_clicks}")
    if not ctx.triggered:
        return no_update

    prop_id = ctx.triggered[0]['prop_id']
    button_children = no_update
    hyper_parameter = pb2.HyperParameters()

    if "resume-pause-train-btn" in prop_id:
        is_training = resume_pause_clicks % 2
        hyper_parameter.is_training = is_training
        button_children = get_pause_play_button() if is_training else get_pause_play_button()
        hyper_parameter.training_steps_to_do = hyper_param_values[5] if is_training else 0
    else:
        btn_dict = eval(prop_id.split('.')[0])
        param_name = btn_dict['idx']
        if param_name == "batch_size":
            hyper_parameter.batch_size = hyper_param_values[0]
        elif param_name == "checkpooint_frequency":
            hyper_parameter.checkpont_frequency = hyper_param_values[1]
        elif param_name == "eval_frequency":
            hyper_parameter.full_eval_frequency = hyper_param_values[2]
        elif param_name == "experiment_name":
            hyper_parameter.experiment_name = hyper_param_values[3]
        elif param_name == "learning_rate":
            hyper_parameter.learning_rate = hyper_param_values[4]
        elif param_name == "training_left":
            hyper_parameter.training_steps_to_do = hyper_param_values[5]

    stub.ExperimentCommand(pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(
            hyper_parameters=hyper_parameter)))
    return button_children

@app.callback(
    Output('experiment_checklist', 'options', allow_duplicate=True),
    Output('experiment_checklist', 'value', allow_duplicate=True),
    Input('graphss-render-freq', 'n_intervals'),
    prevent_initial_call=True
)
def update_experiments_checklist(_):
    experiment_names = list(ui_state.exp_names)
    options = [{'label': exp, 'value': exp} for exp in experiment_names]
    return options, experiment_names

@app.callback(
    Output("experiment_plots_div", "children"),
    Input("graphss-render-freq", "n_intervals"),
    State("experiment_plots_div", "children")
)
def add_graphs_to_div(_, existing_children):
    graph_names = sorted(ui_state.met_names)
    if len(graph_names) == len(existing_children):
        return existing_children
    if not graph_names:
        return no_update
    return [dcc.Graph(id={"type": "graph", "index": g}) for g in graph_names]

# @app.callback(
#     Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
#     Input("graphss-render-freq", "n_intervals"),
#     State({'type': "graph", "index": MATCH}, "id"),
#     State('experiment_checklist', "value"),
#     prevent_initial_call=True
# )

# def update_graph(_, graph_id, selected_experiments):
#     from plotly import graph_objs as go
#     metric_name = graph_id["index"]
#     data = []
#     for exp in selected_experiments:
#         data.extend(ui_state.get_plots_for_exp_name_metric_name(metric_name, exp))
#     if ui_state.plot_name_2_curr_head_point[metric_name] is not None:
#         pt = ui_state.plot_name_2_curr_head_point[metric_name]
#         data.append(go.Scattergl(x=[pt.x], y=[pt.y], mode='markers', marker=dict(size=16, color='red')))
#     return {'data': data, 'layout': go.Layout(title=metric_name, xaxis={'title': 'Seen Samples'}, yaxis={'title': 'Value'})}

@app.callback(
    Output({'type': "graph", "index": MATCH}, "figure", allow_duplicate=True),
    Input("graphss-render-freq", "n_intervals"),
    State({'type': "graph", "index": MATCH}, "id"),
    State('experiment_checklist', "value"),
    # State("plot-smoothness-slider", "value"),
    prevent_initial_call=True,
)
def update_graph(_, graph_id, checklist):
    print("update_graph", graph_id, checklist)
    global ui_state

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
    # print("update_selection_of_checkpoint", hoverData, clickData, figure)

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

    if clickData:
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



if __name__ == '__main__':
    app.run_server(debug=False, port=8053)
