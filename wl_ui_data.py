import argparse
import base64
import threading
import time
import grpc
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update
from dash.dependencies import Input, Output, State

import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import (
    UIState,
    get_play_button_html_elements,
    get_pause_button_html_elements,
    get_data_tab,  
    get_data_query_input_div
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
    get_data_records="train"
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
                get_data_records="train"
            )
            updated_state = stub.ExperimentCommand(update_request)
            ui_state.update_from_server_state(updated_state)
        except Exception as e:
            print("Error updating UI state:", e)
        time.sleep(5)

threading.Thread(target=refresh_ui_state, daemon=True).start()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "WeightsLab - Dataset Only"

app.layout = html.Div([
    dcc.Interval(id='datatbl-render-freq', interval=10000, n_intervals=0),
    dbc.Row([
        dbc.Col(dbc.Button(
            id='resume-pause-train-btn',
            children=get_play_button_html_elements(),
            color='light',
            n_clicks=0,
            style={'marginBottom': '10px'}
        ), width='auto'),
    ], justify='center'),
    get_data_tab(ui_state)
])
app.config.prevent_initial_callbacks = 'initial_duplicate'

@app.callback(
    Output('resume-pause-train-btn', 'children', allow_duplicate=True),
    Input('resume-pause-train-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_training(n_clicks):
    is_training = n_clicks % 2 == 1
    hyper_parameter = pb2.HyperParameters()
    hyper_parameter.is_training = is_training
    request = pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(
            hyper_parameters=hyper_parameter))
    stub.ExperimentCommand(request)
    time.sleep(0.3)
    return get_pause_button_html_elements() if is_training else get_play_button_html_elements()


@app.callback(
    Output('train-data-table', 'data'),
    Input('datatbl-render-freq', 'n_intervals'),
    State('table-refresh-checkbox', 'value'),
)
def update_train_data_table(_, refresh_checkbox):
    if "refresh_regularly" not in refresh_checkbox:
        return no_update
    return ui_state.samples_df.to_dict('records')


@app.callback(
    Input('run-train-data-query', 'n_clicks'),
    State('train-data-query-input', 'value'),
    State('data-query-input-weight', 'value'),
)
def run_query_on_dataset(_, query, weight):
    if weight is None:
        weight = 1.0

    pause_training()


    query_df = ui_state.samples_df.query(query)
    if weight <= 1.0:
        query_df = query_df.sample(frac=weight)
    else:
        query_df = query_df.sample(n=int(weight))

    deny_samples = pb2.DenySamplesOperation()
    deny_samples.sample_ids.extend(query_df["SampleId"].tolist())
    request = pb2.TrainerCommand(deny_samples_operation=deny_samples)
    response = stub.ExperimentCommand(request)

    print(f"Query: {query}, Weight: {weight}, Response: {response}")
    return no_update


@app.callback(
    Output('train-data-div', 'style', allow_duplicate=True),
    Input('sample-inspect-checkboxes', 'value'),
    State('train-data-div', 'style'),
)
def update_div_width(inspect_flags, old_style):
    base = 45
    delta = (90 - base) // 2
    width = base + len(inspect_flags) * delta
    new_style = dict(old_style)
    new_style.update({
        'width': f'{width}vw',
        'maxWdith': f'{width+2}vw',
    })
    return new_style


@app.callback(
    Output('data-panel-col1', 'children', allow_duplicate=True),
    Input('train-data-table', 'selected_rows'),
    State('train-data-table', 'data'),
    State('sample-inspect-checkboxes', 'value'),
)
def render_selected_sample(selected_rows, data, flags):
    if not selected_rows or not flags:
        return []

    selected_index = selected_rows[-1] 
    total_samples = len(data)

    if selected_index <= 0:
        start = 0
        end = min(10, total_samples)
    elif selected_index >= total_samples - 1:
        end = total_samples
        start = max(0, total_samples - 10)
    else:
        start = max(0, selected_index - 4)
        end = min(total_samples, start + 10)
        if end - start < 10:
            start = max(0, end - 10)

    selected_images = []
    for i in range(start, end):
        sample_id = data[i]["SampleId"]
        response = stub.GetSample(pb2.SampleRequest(sample_id=sample_id, origin="train"))
        img = base64.b64encode(response.data).decode('utf-8')
        selected_images.append(html.Img(src=f'data:image/png;base64,{img}', style={
            'width': '12vw', 'height': '12vh', 'margin': '0.5vh'
        }))

    return html.Div(
        children=selected_images,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "1vh",
            "justifyItems": "center",
            "alignItems": "center",
            "paddingLeft": "1vw",
            "paddingTop": "1vh"
        }
    )


@app.callback(
    Output('train-data-table', 'data', allow_duplicate=True),
    Input('train-data-table', 'data'),
    State('train-data-table', 'data_previous'),
    State('table-refresh-checkbox', 'value')
)
def handle_manual_row_deletion(current_data, prev_data, checkboxes):
    if not prev_data:
        return no_update

    prev_ids = {row["SampleId"] for row in prev_data}
    curr_ids = {row["SampleId"] for row in current_data}
    removed = prev_ids - curr_ids

    if removed:
        pause_training()

        deny_request = pb2.TrainerCommand(
            deny_samples_operation=pb2.DenySamplesOperation(
                sample_ids=list(removed)))
        stub.ExperimentCommand(deny_request)

    if "discard_by_flag_flip" in checkboxes:
        for row in prev_data:
            if row["SampleId"] in removed:
                row["Discarded"] = True
        return prev_data

    return current_data


if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
