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
from flask import Response, request, abort
import hashlib
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
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_directory", type=str, required=True,
                        help="Path to the experiment root directory")
    return parser.parse_args()

args = parse_args()
channel = grpc.insecure_channel('localhost:50052')
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
server = app.server
_IMAGE_CACHE = {}

@server.route("/img/<origin>/<int:sid>")
def serve_img(origin, sid):
    try:
        w = int(request.args.get("w", "128"))
        h = int(request.args.get("h", "128"))
        fmt = request.args.get("fmt", "webp")   # 'webp'|'jpeg'|'png'
        if origin not in ("train","eval"): abort(404)

        key = (origin, sid, w, h, fmt)
        if key in _IMAGE_CACHE:
            data, mime, etag = _IMAGE_CACHE[key]
        else:
            # ask backend for exactly the size you need (you already support this)
            batch = stub.GetSamples(pb2.BatchSampleRequest(
                sample_ids=[sid], origin=origin, resize_width=w, resize_height=h
            ))
            if not batch.samples: abort(404)
            raw_png = batch.samples[0].raw_data or batch.samples[0].data

            # Re-encode to a decode-fast & small format
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(raw_png)).convert("RGB")
            buf = io.BytesIO()
            if fmt == "webp":
                im.save(buf, format="WEBP", quality=78, method=4)
                mime = "image/webp"
            else:
                im.save(buf, format="JPEG", quality=80, optimize=True)
                mime = "image/jpeg"
            data = buf.getvalue()

            etag = hashlib.md5(data).hexdigest()
            _IMAGE_CACHE[key] = (data, mime, etag)

        # honor If-None-Match
        inm = request.headers.get("If-None-Match")
        if inm and inm == etag:
            return Response(status=304)

        resp = Response(data, mimetype=mime)
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        resp.headers["ETag"] = etag
        return resp
    except Exception as e:
        print("img route error:", e)
        abort(404)

def make_grid_skeleton(num_cells, origin, img_size):
    # create empty cells with predictable IDs by index
    cells = []
    for i in range(num_cells):
        cells.append(html.Div([
            html.Img(
                id={'type':'sample-img-el', 'origin': origin, 'slot': i},  # use slot index
                src="", loading="lazy", decoding="async",
                width=img_size, height=img_size,
                style={'width': f'{img_size}px', 'height': f'{img_size}px', 'border':'1px solid #ccc'}
            ),
            html.Div(id={'type':'sample-img-label', 'origin': origin, 'slot': i}, style={'fontSize':'11px', 'textAlign':'center'})
        ], style={'display':'flex','flexDirection':'column','alignItems':'center'}))
    return html.Div(children=cells, id={'type':'grid', 'origin':origin}, style={
        'display':'grid',
        'gridTemplateColumns': f'repeat({isqrt(num_cells)}, 1fr)',
        'gap': '4px'
    })


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

def render_images(ui_state: UIState, stub, sample_ids, origin,
                  discarded_ids=None, selected_ids=None):
    selected_ids = set(selected_ids or [])
    task_type = getattr(ui_state, "task_type", "classification")
    imgs = []
    num_images = len(sample_ids)
    cols = isqrt(num_images) or 1
    rows = cols
    img_size = int(512 / max(cols, rows))

    df = ui_state.samples_df if origin == "train" else ui_state.eval_samples_df
    id_to_loss = {int(r["SampleId"]): r.get("LastLoss", None) for _, r in df.iterrows()}
    discarded_ids = set(discarded_ids or set())

    def base_img_style(sid, is_discarded):
        return {
            'width': f'{img_size}px',
            'height': f'{img_size}px',
            'margin': '0.1vh',
            'border': '1px solid #ccc',
            'boxSizing': 'border-box',
            'objectFit': 'contain',
            'imageRendering': 'auto',
            'opacity': 0.25 if is_discarded else 1.0,
            'transition': 'box-shadow 0.06s, opacity 0.1s',
            'boxShadow': '0 0 0 3px rgba(255,45,85,0.95)' if sid in selected_ids else 'none',
            # small paint/layout hints
            'contentVisibility': 'auto',
            'containIntrinsicSize': f'{img_size}px {img_size}px'
        }

    try:
        if task_type == "classification":
            # Use URL endpoint (no base64 through props)
            for sid in sample_ids:
                sid = int(sid)
                is_discarded = sid in discarded_ids
                img_style = base_img_style(sid, is_discarded)
                url = f"/img/{origin}/{sid}?w={img_size}&h={img_size}&fmt=webp"

                img = html.Img(
                    id={'type': 'sample-img-el', 'origin': origin, 'sid': sid},
                    src=url,
                    width=img_size,
                    height=img_size,
                    style=img_style,
                    n_clicks=0
                )

                clickable = html.Div(
                    label_below_img(img, id_to_loss.get(sid, None), img_size),
                    id={'type': 'sample-img', 'origin': origin, 'sid': sid},
                    n_clicks=0,
                    style={'cursor': 'pointer'}
                )
                imgs.append(clickable)

        else:
            # Segmentation: still fetch mask/pred once; render with fixed dims
            with ScopeTimer('getSamples grpc call') as grpc_call:
                batch_response = stub.GetSamples(pb2.BatchSampleRequest(
                    sample_ids=sample_ids,
                    origin=origin,
                    resize_width=img_size,
                    resize_height=img_size
                ))
            print(grpc_call)

            for sample in batch_response.samples:
                sid = int(sample.sample_id)
                is_discarded = sid in discarded_ids
                last_loss = id_to_loss.get(sid, None)

                input_b64 = base64.b64encode(sample.raw_data).decode('utf-8')
                gt_mask_b64 = base64.b64encode(sample.mask).decode('utf-8') if sample.mask else ""
                pred_mask_b64 = base64.b64encode(sample.prediction).decode('utf-8') if sample.prediction else ""

                def png_img(src_b64, border):
                    return html.Img(
                        src=f'data:image/png;base64,{src_b64}',
                        width=img_size,
                        height=img_size,
                        style={'width': f'{img_size}px', 'height': f'{img_size}px', 'border': border,
                               'contentVisibility': 'auto', 'containIntrinsicSize': f'{img_size}px {img_size}px'}
                    )

                input_img_div = html.Div([
                    html.Img(
                        src=f'data:image/png;base64,{input_b64}',
                        width=img_size,
                        height=img_size,
                        style={'width': f'{img_size}px', 'height': f'{img_size}px', 'border': '1px solid #888',
                               'contentVisibility': 'auto', 'containIntrinsicSize': f'{img_size}px {img_size}px'}
                    ),
                    html.Div(
                        f"ID: {sid}", style={
                            'position': 'absolute', 'top': '2px', 'left': '4px',
                            'background': 'rgba(0,0,0,0.55)', 'color': 'white',
                            'fontSize': '10px', 'padding': '1px 5px', 'borderRadius': '3px'
                        }
                    ),
                    html.Div(
                        f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -",
                        style={
                            'position': 'absolute', 'top': '2px', 'right': '4px',
                            'background': 'rgba(0,0,0,0.55)', 'color': 'white',
                            'fontSize': '10px', 'padding': '1px 5px', 'borderRadius': '3px'
                        }
                    ),
                    html.Div("Input", style={'fontSize': 10, 'textAlign': 'center'})
                ], style={'position': 'relative', 'display': 'inline-block',
                          'width': f'{img_size}px', 'height': f'{img_size}px'})

                triplet = html.Div([
                    input_img_div,
                    html.Div([png_img(gt_mask_b64, '1px solid green'),
                              html.Div("Target", style={'fontSize': 10, 'textAlign': 'center'})]),
                    html.Div([png_img(pred_mask_b64, '1px solid blue'),
                              html.Div("Prediction", style={'fontSize': 10, 'textAlign': 'center'})]),
                ], style={
                    'display': 'flex',
                    'flexDirection': 'row',
                    'gap': '4px',
                    'marginBottom': '8px',
                    'border': '4px solid red' if sid in selected_ids else 'none',
                    'transition': 'border 0.3s, opacity 0.3s',
                    'opacity': 0.25 if is_discarded else 1.0
                })

                clickable = html.Div(
                    [triplet],
                    id={'type': 'sample-img', 'origin': origin, 'sid': sid},
                    n_clicks=0,
                    style={'cursor': 'pointer'}
                )
                imgs.append(clickable)

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
        selected_rows=[],
        style_data_conditional=[],
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
        selected_rows=[],
        style_data_conditional=[],
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
        dbc.Col( 
            dbc.Button(
                "Progressive Chunk Update", id='progressive-chunk-train-btn',
                color='success', 
                n_clicks=0, 
                style={'width': '12vw', 'marginLeft': '1vw'}
            )
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
        dbc.Col( 
            dbc.Button(
                "Progressive Chunk Update", id='progressive-chunk-eval-btn',
                color='success', 
                n_clicks=0, 
                style={'width': '12vw', 'marginLeft': '1vw'}
            )
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
    dcc.Store(id='train-image-selected-ids', data=[]),
    dcc.Store(id='eval-image-selected-ids', data=[]),
    dcc.Interval(id='datatbl-render-freq', interval=5000, n_intervals=0),
    get_header_hyper_params_div(ui_state),
    get_data_tab(ui_state),
])

def progressive_chunk_update_v2(ui_state, stub, chunk_weight):
    df = ui_state.samples_df.copy()
    current_train = df[~df['Discarded']]
    prev_batch = current_train[current_train['LastLoss'] != -1]
    untrained = df[(df['Discarded']) & (df['LastLoss'] == -1)]

    misclassified = prev_batch[prev_batch['Prediction'] != prev_batch['Target']]
    correct = prev_batch[prev_batch['Prediction'] == prev_batch['Target']]

    keep_correct = correct.sample(frac=0.5) if not correct.empty else correct

    available_ids = untrained['SampleId'].tolist()
    if len(available_ids) == 0:
        return "No more new samples to add."

    if isinstance(chunk_weight, float) and 0 < chunk_weight < 1:
        n_new = int(len(available_ids) * chunk_weight)
    else:
        n_new = int(chunk_weight)
    n_new = max(1, min(n_new, len(available_ids)))
    new_chunk = random.sample(available_ids, n_new)

    keep_ids = (
        misclassified['SampleId'].tolist() +
        keep_correct['SampleId'].tolist() +
        new_chunk
    )

    deny_ids = [sid for sid in current_train['SampleId'] if sid not in keep_ids]
    allow_ids = keep_ids

    if deny_ids:
        deny_op = pb2.DenySamplesOperation(sample_ids=deny_ids, accumulate=False)
        stub.ExperimentCommand(pb2.TrainerCommand(deny_samples_operation=deny_op))
    if allow_ids:
        allow_op = pb2.DenySamplesOperation(sample_ids=allow_ids, accumulate=False)
        stub.ExperimentCommand(pb2.TrainerCommand(remove_from_denylist_operation=allow_op))

    return (f"Added {len(new_chunk)} new, kept {len(misclassified)} misclassified, "
            f"{len(keep_correct)} correct (50%), denied {len(deny_ids)}")

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
    Output('train-sample-panel', 'children'),
    Output('eval-sample-panel', 'children'),
    Input('train-data-table', 'derived_viewport_data'),
    Input('eval-data-table', 'derived_viewport_data'),
    Input('sample-inspect-checkboxes', 'value'),
    Input('eval-sample-inspect-checkboxes', 'value'),
    Input('data-tabs', 'value'),
    State('train-image-selected-ids', 'data'), 
    State('eval-image-selected-ids', 'data'),  
    prevent_initial_call=True
)
def render_samples(
    train_viewport, eval_viewport, train_flags, eval_flags, tab,
    train_selected_ids, eval_selected_ids
):
    with ScopeTimer('Complete render samples callback') as complete:
        panels = [no_update, no_update]

        if tab == 'train' and 'inspect_sample_on_click' in (train_flags or []) and train_viewport:
            ids = [row['SampleId'] for row in train_viewport]
            discarded_ids = set(ui_state.samples_df.loc[ui_state.samples_df['Discarded'], 'SampleId'])
            with ScopeTimer('Render_images call') as render_images_t:
                panels[0] = render_images(ui_state, stub, ids, origin='train',
                                        discarded_ids=discarded_ids,
                                        selected_ids=(train_selected_ids or []))
            print(render_images_t)


        elif tab == 'eval' and 'inspect_sample_on_click' in (eval_flags or []) and eval_viewport:
            ids = [row['SampleId'] for row in eval_viewport]
            discarded_ids = set(ui_state.eval_samples_df.loc[ui_state.eval_samples_df['Discarded'], 'SampleId'])
            panels[1] = render_images(ui_state, stub, ids, origin='eval',
                                    discarded_ids=discarded_ids,
                                    selected_ids=(eval_selected_ids or []))
    print(complete)
    return panels


@app.callback(
    Output({'type': 'sample-img-el', 'origin': 'train', 'slot': ALL}, 'src'),
    Output({'type': 'sample-img-label', 'origin': 'train', 'slot': ALL}, 'children'),
    Input('train-data-table', 'derived_viewport_data'),
    State('grid-preset-dropdown', 'value'),
    prevent_initial_call=True
)
def update_train_slots(viewport_rows, grid_count):
    n = grid_count
    urls = [""] * n
    labels = [""] * n
    if not viewport_rows:
        return urls, labels

    rows = viewport_rows[:n]
    img_size = int(512 / max(isqrt(n) or 1, isqrt(n) or 1))
    for i, row in enumerate(rows):
        sid = row['SampleId']
        last_loss = row.get('LastLoss', None)
        urls[i] = f"/img/train/{sid}?w={img_size}&h={img_size}&fmt=webp"
        labels[i] = f"Loss: {last_loss:.4f}" if last_loss is not None else "Loss: -"
    return urls, labels


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


@app.callback(
    Output('train-image-selected-ids', 'data', allow_duplicate=True),
    Output('eval-image-selected-ids', 'data', allow_duplicate=True),
    Input({'type': 'sample-img-click', 'sample_id': ALL, 'origin': ALL}, 'n_clicks'),
    State('train-image-selected-ids', 'data'),
    State('eval-image-selected-ids', 'data'),
    prevent_initial_call=True
)
def toggle_image_selection(clicks, train_ids, eval_ids):
    triggered = ctx.triggered_id
    if not triggered or 'sample_id' not in triggered:
        return dash.no_update, dash.no_update

    sid = triggered['sample_id']
    origin = triggered['origin']
    train_ids = train_ids or []
    eval_ids = eval_ids or []

    if origin == "train":
        if sid in train_ids:
            train_ids = [x for x in train_ids if x != sid]
        else:
            train_ids = train_ids + [sid]
        return train_ids, dash.no_update
    elif origin == "eval":
        if sid in eval_ids:
            eval_ids = [x for x in eval_ids if x != sid]
        else:
            eval_ids = eval_ids + [sid]
        return dash.no_update, eval_ids
    return dash.no_update, dash.no_update


@app.callback(
    Output('train-data-table', 'selected_rows', allow_duplicate=True),
    Output('eval-data-table', 'selected_rows', allow_duplicate=True),
    Input('train-image-selected-ids', 'data'),
    Input('eval-image-selected-ids', 'data'),
    State('train-data-table', 'data'),
    State('eval-data-table', 'data'),
    prevent_initial_call=True
)
def update_table_selection_from_store(train_selected_ids, eval_selected_ids, train_data, eval_data):
    train_idx = []
    eval_idx = []
    if train_selected_ids:
        train_idx = [i for i, row in enumerate(train_data) if row["SampleId"] in train_selected_ids]
    if eval_selected_ids:
        eval_idx = [i for i, row in enumerate(eval_data) if row["SampleId"] in eval_selected_ids]
    return train_idx, eval_idx

@app.callback(
    Output('train-data-table', 'style_data_conditional'),
    Input('train-data-table', 'selected_rows'),
    State('train-data-table', 'data')
)
def highlight_selected_rows(selected_rows, data):
    if not selected_rows:
        return []
    filter_query = ' || '.join([f'{{SampleId}} = {data[i]["SampleId"]}' for i in selected_rows])
    return [{
        "if": {"filter_query": filter_query},
        "backgroundColor": "#ffe6b3",
        "fontWeight": "bold"
    }]


@app.callback(
    Output('train-image-selected-ids', 'data', allow_duplicate=True),
    Output('eval-image-selected-ids', 'data', allow_duplicate=True),
    Input('train-data-table', 'selected_rows'),
    Input('eval-data-table', 'selected_rows'),
    State('train-data-table', 'data'),
    State('eval-data-table', 'data'),
    prevent_initial_call=True
)
def store_highlighted_samples(train_selected_rows, eval_selected_rows, train_data, eval_data):
    train_ids = []
    eval_ids = []
    if train_selected_rows and train_data:
        train_ids = [train_data[i]['SampleId'] for i in train_selected_rows]
    if eval_selected_rows and eval_data:
        eval_ids = [eval_data[i]['SampleId'] for i in eval_selected_rows]

    if not train_selected_rows:
        train_ids = []
    if not eval_selected_rows:
        eval_ids = []
        
    return train_ids, eval_ids


if __name__ == '__main__':
    app.run_server(debug=False, port=8052)
