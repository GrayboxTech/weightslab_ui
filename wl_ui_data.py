import argparse
import base64
import threading
import time
import grpc
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash.dash_table.Format import Format, Scheme

import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import (
    UIState,
    get_play_button_html_elements,
    get_pause_button_html_elements,
    get_data_query_input_div,
    _DISPLAY_COLUMNS,
)
from scope_timer import ScopeTimer
update_timer = ScopeTimer(tag="update_train_data_table")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True,
                        help="Path to the experiment root directory")
    return parser.parse_args()

args = parse_args()
channel = grpc.insecure_channel('localhost:50051')
stub = pb2_grpc.ExperimentServiceStub(channel)
ui_state = UIState(args.root_directory)
initial_state = stub.ExperimentCommand(
    pb2.TrainerCommand(
        get_hyper_parameters=True,
        get_interactive_layers=False,
        get_data_records="train"
    )
)
ui_state.update_from_server_state(initial_state)


def pause_training():
    hyper = pb2.HyperParameters(is_training=False)
    cmd = pb2.TrainerCommand(
        hyper_parameter_change=pb2.TrainerCommand(
            hyper_parameters=hyper
        )
    )
    stub.ExperimentCommand(cmd)
    time.sleep(0.1)


def refresh_ui_state():
    while True:
        try:
            req = pb2.TrainerCommand(
                get_hyper_parameters=True,
                get_interactive_layers=False,
                get_data_records="train"
            )
            state = stub.ExperimentCommand(req)
            ui_state.update_from_server_state(state)
        except Exception as e:
            print("Error updating UI state:", e)
        time.sleep(5)


threading.Thread(target=refresh_ui_state, daemon=True).start()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.title = "WeightsLab - Dataset Only"
app.config.prevent_initial_callbacks = 'initial_duplicate'

# dropdown for grid preset: display counts 9,16,25,36
grid_preset_dropdown = dcc.Dropdown(
    id='grid-preset-dropdown',
    options=[
        {'label': str(x * x), 'value': x * x} for x in [3, 4, 5, 6]
    ],
    value=16,
    clearable=False,
    style={'width': '6vw'}
)

def get_data_tab(ui_state: UIState):
    cols = []
    for column in _DISPLAY_COLUMNS:
        if column == "Encounters":
            continue
        spec = {"name": column, "id": column,
                "type": "text" if column == "Discarded" else 'any'}
        if column == "LastLoss":
            spec["format"] = Format(precision=2, scheme=Scheme.fixed)
        cols.append(spec)

    table = dash_table.DataTable(
        id='train-data-table',
        data=ui_state.samples_df.to_dict('records'),
        columns=cols,
        sort_action="native",
        page_action="native",
        page_size=16,
        row_selectable='multi',
        row_deletable=True,
        editable=True,
        virtualization=True,
        style_table={
            'height': '25vh',
            'overflowY': 'auto',
            'width': '38vw',
            'margin': '2px',
            'padding': '2px'
        },
        style_cell={'textAlign': 'left', 'minWidth': '4vw', 'maxWidth': '4.5vw'}
    )

    controls = html.Div([
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

    left = html.Div([
        html.H2("Train Dataset"),
        controls,
        table,
        get_data_query_input_div(ui_state)
    ])

    right = html.Div(id='data-panel-col1', children=[],
                style={'overflow': 'auto'})

    return html.Div(
        dbc.Row([
            dbc.Col(left, width=8),
            dbc.Col(right, width=4)
        ], align='start'),
        style={
            'margin': '4vw', 'padding': '2vw',
            'borderRadius': '15px', 'border': '2px solid #666',
            'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
            'width': '87vw'
        }
    )


app.layout = html.Div([
    dcc.Interval(id='datatbl-render-freq', interval=10000, n_intervals=0),
    dbc.Row([
        dbc.Col(dbc.Button(
            id='resume-pause-train-btn',
            children=get_play_button_html_elements(),
            color='light', n_clicks=0,
            style={'marginBottom': '10px'}
        ), width='auto')
    ], justify='center'),
    get_data_tab(ui_state)
])


@app.callback(
    Output('resume-pause-train-btn', 'children', allow_duplicate=True),
    Input('resume-pause-train-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_training(n_clicks):
    is_training = n_clicks % 2 == 1
    cmd = pb2.TrainerCommand(
    hyper_parameter_change=pb2.HyperParameterCommand(
        hyper_parameters=pb2.HyperParameters(is_training=is_training)
    ))
    stub.ExperimentCommand(cmd)
    time.sleep(0.1)
    return get_pause_button_html_elements() if is_training else get_play_button_html_elements()


@app.callback(
    Output('train-data-table', 'data'),
    Input('datatbl-render-freq', 'n_intervals'),
    State('table-refresh-checkbox', 'value')
)
def update_train_data_table(_, chk):
    #with ScopeTimer(tag="update_train_data_table") as timer:
    if 'refresh_regularly' not in chk:
        return no_update
    with update_timer:
        return ui_state.samples_df.to_dict('records')


@app.callback(
    Output('train-data-table', 'page_size'),
    Input('grid-preset-dropdown', 'value')
)
def update_page_size(grid_count):
    return grid_count


@app.callback(
    Output('data-panel-col1', 'children', allow_duplicate=True),
    Input('train-data-table', 'derived_virtual_data'),
    Input('train-data-table', 'derived_viewport_indices'),
    Input('grid-preset-dropdown', 'value'),
    Input('train-data-table', 'selected_rows')
)
def render_visible_samples(virtual_data, viewport_indices, count, selected_rows):
    if not virtual_data or not viewport_indices:
        return []
    from math import isqrt
    cols = rows = isqrt(count)
    imgs = []
    for i in viewport_indices[:count]:
        row = virtual_data[i]
        sid = row['SampleId']
        res = stub.GetSample(pb2.SampleRequest(sample_id=sid, origin='train'))
        b64 = base64.b64encode(res.data).decode('utf-8')
        border = '4px solid red' if selected_rows and i in selected_rows else '1px solid #ccc'
        imgs.append(html.Img(src=f'data:image/png;base64,{b64}', style={
            'width': 'auto', 'height': 'auto', 'margin': '0.1vh', 'border': border
        }))
    return html.Div(children=imgs, style={
        'display': 'grid',
        'gridTemplateColumns': f'repeat({cols}, 1fr)',
        'columnGap': '0.1vw',
        'rowGap': '0.1vh',
        'justifyItems': 'center',
        'alignItems': 'center',
        'paddingLeft': '0.25vw'
    })


@app.callback(
    Output('train-data-table', 'data', allow_duplicate=True),
    Input('train-data-table', 'data'),
    State('train-data-table', 'data_previous'),
    State('table-refresh-checkbox', 'value')
)
def handle_manual_row_deletion(current_data, prev_data, chk):
    if not prev_data:
        return no_update
    prev_ids = {r['SampleId'] for r in prev_data}
    curr_ids = {r['SampleId'] for r in current_data}
    removed = prev_ids - curr_ids
    if removed:
        pause_training()
        stub.ExperimentCommand(
            pb2.TrainerCommand(deny_samples_operation=pb2.DenySamplesOperation(
                sample_ids=list(removed)
            ))
        )
    if 'discard_by_flag_flip' in chk:
        for r in prev_data:
            if r['SampleId'] in removed:
                r['Discarded'] = True
        return prev_data
    return current_data


if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
