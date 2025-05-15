import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, ctx, no_update
from dash.dependencies import Input, Output, State, ALL, MATCH
import threading
import grpc
import time
import argparse
import collections


import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import (
    UIState, get_weights_div,
    convert_checklist_to_df_head, format_values_df,
    get_layer_headings, layer_div_width, get_play_button_html_elements,
    get_pause_button_html_elements,
    get_hyper_params_div,

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
        time.sleep(5)

thread = threading.Thread(target=refresh_ui_state, daemon=True)
thread.start()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "WeightsLab - Architecture Only"

app.layout = html.Div([
    dcc.Interval(id='weights-render-freq', interval=5000, n_intervals=0),
    html.H1("Model Architecture", style={'textAlign': 'center'}),
    get_hyper_params_div(ui_state),
    get_weights_div(ui_state),
])
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
            button_children = get_pause_button_html_elements()
            hyper_parameter.training_steps_to_do = hyper_param_values[5]
        else:
            button_children = get_play_button_html_elements()
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
#     Output('resume-pause-train-btn', 'children', allow_duplicate=True),
#     Input('resume-pause-train-btn', 'n_clicks'),
#     prevent_initial_call=True
# )
# def toggle_training(n_clicks):
#     is_training = n_clicks % 2 == 1
#     hyper_parameter = pb2.HyperParameters()
#     hyper_parameter.is_training = is_training
#     request = pb2.TrainerCommand(
#         hyper_parameter_change=pb2.HyperParameterCommand(
#             hyper_parameters=hyper_parameter))
#     stub.ExperimentCommand(request)
#     return get_pause_button_html_elements() if is_training else get_play_button_html_elements()

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
    Input({'type': 'layer-add-btn', 'layer_id': ALL}, 'n_clicks')
)
def add_neuron(n_clicks):
    if not ctx.triggered: return dash.no_update
    triggered = ctx.triggered_id
    if not triggered: return dash.no_update

    layer_id = triggered['layer_id']
    pause_training()
    op = pb2.WeightOperation(
        op_type=pb2.WeightOperationType.ADD_NEURONS,
        layer_id=layer_id,
        neurons_to_add=1
    )
    stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=op))
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
    pause_training()
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
    )
def run_query_on_neurons(_, query, weight, action):
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

    # print("Selected neurons:", selected_neurons_df)
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


if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
