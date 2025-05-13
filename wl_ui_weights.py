import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, ctx
from dash.dependencies import Input, Output, State, ALL, MATCH
import threading
import grpc
import time
import argparse

import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import (
    UIState, get_weights_div,
    convert_checklist_to_df_head, format_values_df,
    get_layer_headings, layer_div_width
)

# ROOT_DIR = "./fashion-mnist-dev7/"


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
    get_weights_div(ui_state),
])

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

    _, sub_heading = get_layer_headings(layer_row)
    layer_width = layer_div_width(checklist_values)
    layer_div_style['minWidth'] = f"{layer_width}px"

    return columns, records, style_data_conditional, layer_div_style, sub_heading

@app.callback(
    Output({'type': 'layer-add-btn', 'layer_id': ALL}, 'n_clicks'),
    Input({'type': 'layer-add-btn', 'layer_id': ALL}, 'n_clicks')
)
def add_neuron(n_clicks):
    if not ctx.triggered: return dash.no_update
    triggered = ctx.triggered_id
    if not triggered: return dash.no_update

    layer_id = triggered['layer_id']
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
    State('neuron-action-dropdown', 'value')
)
def handle_query(n_clicks, query, weight, action):
    if not query or not action:
        return

    if weight is None:
        weight = 1.0

    try:
        # Keep MultiIndex for query to work on layer_id and neuron_id
        df = ui_state.get_neurons_df()
        selected_df = df.query(query)

        # Sample the neurons based on the weight
        if weight <= 1.0:
            selected_df = selected_df.sample(frac=weight)
        else:
            selected_df = selected_df.sample(n=int(weight))

        # Reset index only after query for iteration
        selected_df = selected_df.reset_index()

        # Action: highlight (UI only)
        if action == 'highlight':
            for _, row in selected_df.iterrows():
                ui_state.selected_neurons[row['layer_id']].append(row['neuron_id'])
            return

        # Map action to protobuf operation
        op_type = dict(
            delete=pb2.WeightOperationType.REMOVE_NEURONS,
            reinitialize=pb2.WeightOperationType.REINITIALIZE,
            freeze=pb2.WeightOperationType.FREEZE
        ).get(action)

        if not op_type:
            return

        # Construct and send the operation
        op = pb2.WeightOperation(
            op_type=op_type,
            neuron_ids=[
                pb2.NeuronId(layer_id=row['layer_id'], neuron_id=row['neuron_id'])
                for _, row in selected_df.iterrows()
            ]
        )
        stub.ManipulateWeights(pb2.WeightsOperationRequest(weight_operation=op))

    except Exception as e:
        print(f"Query Error: {e}")


if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
