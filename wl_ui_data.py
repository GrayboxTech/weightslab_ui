import argparse
import base64
import threading
import time
import grpc
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, no_update, ctx, dash_table
from dash.dependencies import Input, Output, State, ALL
from dash.dash_table.Format import Format, Scheme
from math import isqrt
import re
import experiment_service_pb2 as pb2
import experiment_service_pb2_grpc as pb2_grpc
from weights_lab import (
    UIState,
    get_pause_play_button,
    get_header_hyper_params_div,
    _DISPLAY_COLUMNS,
)
from scope_timer import ScopeTimer
import numpy as np
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
        # get_data_records="train"
    )
)
ui_state.update_from_server_state(initial_state)

def refresh_ui_state():
    while True:
        try:
            for dataset in ["train", "eval"]:
                req = pb2.TrainerCommand(
                    get_hyper_parameters=True,
                    get_interactive_layers=False,
                    get_data_records=dataset
                )
                state = stub.ExperimentCommand(req)
                ui_state.update_from_server_state(state)
        except Exception as e:
            print("Error updating UI state:", e)


threading.Thread(target=refresh_ui_state, daemon=True).start()

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.ZEPHYR])
app.config.suppress_callback_exceptions = True
app.title = "WeightsLab - Dataset Only"
app.config.prevent_initial_callbacks = 'initial_duplicate'

# dropdown for grid preset: display counts 9,16,25,36
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

def render_segmentation_triplet(input_b64, gt_mask_b64, pred_mask_b64, is_selected, img_size, is_discarded):
    return html.Div([
        html.Div([
            html.Img(src=f'data:image/png;base64,{input_b64}', style={'width':f'{img_size}px','border':'1px solid #888'}),
            html.Div("Input", style={'fontSize':10, 'textAlign':'center'})
        ]),
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



def render_images(sample_ids, selected_ids, origin, discarded_ids=None):
    task_type = getattr(ui_state, "task_type", "classification")
    imgs = []
    num_images = len(sample_ids)
    cols = isqrt(num_images) or 1
    rows = cols
    img_size = int(512 / max(cols, rows))

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
                imgs.append(render_segmentation_triplet(input_b64, gt_mask_b64, pred_mask_b64, is_selected, img_size, is_discarded))

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
                imgs.append(html.Img(
                    src=f'data:image/png;base64,{b64}',
                    style=style
                ))
    except Exception as e:
        print(f"[ERROR] {origin} sample rendering failed: {e}")
        return no_update


    return html.Div(children=imgs, style={
        'display': 'grid',
        'gridTemplateColumns': f'repeat({cols}, 1fr)',
        'columnGap': '0.1vw',
        'rowGap': '0.1vh',
        'width': '512px',
        'height': '512px',
        'justifyItems': 'center',
        'alignItems': 'center',
        'paddingLeft': '0.01vw'
    })

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

def rewrite_query_for_lists(query, task_type):
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

def get_query_context(tab_type):
    if tab_type == "train":
        df = ui_state.samples_df.copy()
        remove_op = "remove_from_denylist_operation"
        deny_op = "deny_samples_operation"
    else:
        df = ui_state.eval_samples_df.copy()
        remove_op = "remove_eval_from_denylist_operation"
        deny_op = "deny_eval_samples_operation"
    return df, remove_op, deny_op

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

    train_table = dash_table.DataTable(
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
            'height': 'auto',
            'overflowY': 'auto',
            'width': '35vw',
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
        row_deletable=True,
        editable=True,
        virtualization=True,
        style_table={
            'height': 'auto',
            'overflowY': 'auto',
            'width': '35vw',
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
                    html.H2("Train Dataset"),
                    train_controls,
                    html.Div([
                        html.Div([train_table], style={
                            'flex': '0 0 35vw', 
                            'minWidth': '35vw'
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
                            'minWidth': '35vw'
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

app.layout = html.Div([
    dcc.Interval(id='datatbl-render-freq', interval=5000, n_intervals=0),
    get_header_hyper_params_div(ui_state),
    get_data_tab(ui_state),
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
    Output('train-data-table', 'data'),
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
    return df.to_dict('records')



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

    if tab == 'train' and 'inspect_sample_on_click' in train_flags and train_viewport:
        df = ui_state.samples_df
        ids = [row['SampleId'] for row in train_viewport if row['SampleId'] in df['SampleId'].values]
        selected_ids = set(df.iloc[i]['SampleId'] for i in train_selected_rows or [])
        discarded_ids = set(df.loc[df['Discarded'], 'SampleId'])
        panels[0] = render_images(ids, selected_ids, origin='train', discarded_ids=discarded_ids)
    elif tab == 'eval' and 'inspect_sample_on_click' in eval_flags and eval_viewport:
        df = ui_state.eval_samples_df
        ids = [row['SampleId'] for row in eval_viewport if row['SampleId'] in df['SampleId'].values]
        selected_ids = set(df.iloc[i]['SampleId'] for i in eval_selected_rows or [])
        discarded_ids = set(df.loc[df['Discarded'], 'SampleId'])
        panels[1] = render_images(ids, selected_ids, origin='eval', discarded_ids=discarded_ids)

    return panels


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
        df, remove_op_key, deny_op_key = get_query_context(tab_type)
        task_type = getattr(ui_state, "task_type", "classification")
        if task_type == "classification":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda v: v[0] if isinstance(v, (list, np.ndarray)) and len(v) == 1 else v)
        elif task_type == "segmentation":
            for col in ["Prediction", "Target"]:
                if col in df.columns:
                    df[col + "ClassesStr"] = df[col].apply(lambda arr: ",".join([str(x) for x in arr]))
        filter_fn = rewrite_query_for_lists(query, task_type)
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
    Output('train-data-table', 'data', allow_duplicate=True),
    Input('train-data-table', 'data'),
    State('train-data-table', 'data_previous'),
    State('table-refresh-checkbox', 'value')
)
def handle_manual_train_row_deletion(current_data, prev_data, chk):
    if prev_data is None:
        return no_update
    prev_ids = {r['SampleId'] for r in prev_data}
    curr_ids = {r['SampleId'] for r in current_data}
    removed = prev_ids - curr_ids
    if removed:
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


@app.callback(
    Output('eval-data-table', 'data', allow_duplicate=True),
    Input('eval-data-table', 'data'),
    State('eval-data-table', 'data_previous'),
    State('table-refresh-checkbox', 'value')
)
def handle_manual_eval_row_deletion(current_data, prev_data, chk):
    if not prev_data:
        return no_update
    prev_ids = {r['SampleId'] for r in prev_data}
    curr_ids = {r['SampleId'] for r in current_data}
    removed = prev_ids - curr_ids
    if removed:
        stub.ExperimentCommand(
            pb2.TrainerCommand(deny_eval_samples_operation=pb2.DenySamplesOperation(
                sample_ids=list(removed)
            ))
        )
    if 'discard_by_flag_flip' in chk:
        for r in prev_data:
            if r['SampleId'] in removed:
                r['Discarded'] = True
        return prev_data
    return current_data


@app.callback(
    Output('train-data-table', 'selected_rows'),
    Output('eval-data-table', 'selected_rows'),
    Input({'type': 'sample-img-click', 'sample_id': ALL, 'origin': ALL}, 'n_clicks'),
    State('train-data-table', 'data'),
    State('eval-data-table', 'data'),
    prevent_initial_call=True
)
def update_selection_from_image_click(all_clicks, train_data, eval_data):
    if not any(all_clicks):
        return dash.no_update, dash.no_update

    # Get the triggered image ID
    triggered = ctx.triggered_id
    if not triggered or "sample_id" not in triggered:
        return dash.no_update, dash.no_update

    sid = triggered["sample_id"]
    origin = triggered["origin"]

    if origin == "train":
        idx = next((i for i, row in enumerate(train_data) if row["SampleId"] == sid), None)
        return [idx] if idx is not None else [], dash.no_update

    elif origin == "eval":
        idx = next((i for i, row in enumerate(eval_data) if row["SampleId"] == sid), None)
        return dash.no_update, [idx] if idx is not None else []

    return dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
