import argparse
import base64
import threading
import time
import grpc
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import UIState

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
    get_data_records="train"
))
ui_state.update_from_server_state(initial_state)

# Background thread to keep refreshing ui_state
def refresh_ui_state():
    while True:
        try:
            update_request = pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=True,
                get_data_records="train"
            )
            updated_state = stub.ExperimentCommand(update_request)
            ui_state.update_from_server_state(updated_state)
        except Exception as e:
            print("Error updating UI state:", e)
        time.sleep(5)

thread = threading.Thread(target=refresh_ui_state, daemon=True)
thread.start()

# Dash App 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "WeightsLab - Dataset Only"

app.layout = html.Div([
    dcc.Interval(id='datatbl-render-freq', interval=10000, n_intervals=0),
    html.H1("Train Dataset", style={'textAlign': 'center'}),
    dcc.Checklist(
        id='sample-inspect-checkboxes',
        options=[{'label': 'Inspect on click', 'value': 'inspect_sample_on_click'}],
        value=[],
        inline=True,
        style={'marginBottom': '10px'}
    ),
    dash_table.DataTable(
        id='train-data-table',
        columns=[{"name": i, "id": i} for i in ui_state.samples_df.columns],
        data=ui_state.samples_df.to_dict('records'),
        row_selectable='single',
        style_table={'overflowY': 'scroll', 'height': '400px'},
        page_size=20,
    ),
    html.Div(id='data-panel-col1', children=[]),
])

@app.callback(
    Output('train-data-table', 'data'),
    Input('datatbl-render-freq', 'n_intervals')
)
def update_data_table(_):
    return ui_state.samples_df.to_dict('records')

@app.callback(
    Output('data-panel-col1', 'children'),
    Input('train-data-table', 'selected_rows'),
    State('train-data-table', 'data'),
    State('sample-inspect-checkboxes', 'value'),
)
def render_data_sample(selected_rows, data, inspect_checkboxes):
    if not selected_rows or not inspect_checkboxes:
        return []
    selected_row_index = selected_rows[-1]
    row = data[selected_row_index]
    selected_sample_id = row["SampleId"]

    request = pb2.SampleRequest(sample_id=selected_sample_id, origin="train")
    response = stub.GetSample(request)
    image_base64 = base64.b64encode(response.data).decode('utf-8')

    return html.Img(
        src=f'data:image/png;base64,{image_base64}',
        style={'width': '24vw', 'height': '24vh', 'margin': '1vh'}
    )

if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
