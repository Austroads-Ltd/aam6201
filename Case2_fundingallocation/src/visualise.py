"""
Plotting utils for notebook
"""

import numpy as np
import plotly.graph_objects as go

def plot_output_plane(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray,
    color: np.ndarray=None,
    x_label: str='X-Axis',
    y_label: str='Y-Axis',
    z_label: str='Z-Axis',
    make_fig: bool=True,
    colorbar: dict=None,
    coloraxis: str=None,
    **kwargs
):
    """
    Plot 3d plot given values of budget splits for 2 criteria on the x and y axis, 
    as well as their corresponding level of service on z axis.

    If make_fig is set to False, return the scatter data instead.
    """    
    assert x.ndim == 1 and y.ndim == 1 and z.ndim == 1 
    assert len(y) == len(x) and len(y) == len(z)

    
    data = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            color=(color if color is not None else z), colorbar=colorbar, coloraxis=coloraxis,
        ),
        **kwargs
    )

    if make_fig:
        fig = go.Figure(data=[data])
        fig.update_layout(
            autosize=False,
            width=1000,
            height=800,
            title='Level of Service against Splits',
            scene=dict(
                yaxis_title=y_label,
                yaxis_tickfontsize=8,
                xaxis_title=x_label,
                xaxis_tickfontsize=8,
                zaxis_title=z_label,
                zaxis_tickfontsize=8,
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        return fig
    
    else:
        return data

def update_scenelabels(fig: go.Figure, row: int=1, col: int=1, fontsize: int=10, x_label: str='X-Axis', y_label: str='Y-Axis', z_label: str='Z-Axis'):
    """
    Update axis labels
    """
    fig.update_scenes(
        row=row, col=col,
        xaxis=dict(
            title_text=x_label,
            title_font_size=fontsize,
            tickfont_size=fontsize
        ),
        yaxis=dict(
            title_text=y_label,
            title_font_size=fontsize,
            tickfont_size=fontsize
        ),
        zaxis=dict(
            title_text=z_label,
            title_font_size=fontsize,
            tickfont_size=fontsize
        ),
    )

def add_2d_plane(fig: go.Figure, z: float,
    row: int=1, col: int=1, 
    x_range: tuple=(0, 1), 
    y_range: tuple=(0, 1),
    name=None
):
    """Add valid region to a plot"""
    # 11 points
    x = np.linspace(x_range[0], x_range[1], num=10) 
    y = np.linspace(y_range[0], y_range[1], num=10) 
    z = np.ones((len(x), len(y))) * z

    trace = \
        go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, 'rgb(255,0,0)'], [1, 'rgb(255,0,0)']],
            showscale=False,
            opacity=0.4,
            name=name
        )
    
    fig.add_trace(trace, row=row, col=col)
    return trace

def add_valid_region_3d(fig: go.Figure,
    z_max_range: tuple,
    row: int=1, col: int=1, 
    x_max_range:tuple=(0, 1), 
    y_max_range: tuple=(0, 1), 
    x_range: tuple=(0, 1), 
    y_range: tuple=(0, 1),
    name=None
):
    # 4 outer cubes:
    x_max_range[0], x_range[0]
    transforms = [
        {
            'x': {0: x_max_range[0], 1: x_range[0]},
            'y': {0: y_range[0], 1: y_range[1]}
        },
        {
            'x': {0: x_range[0], 1: x_range[1]},
            'y': {0: y_max_range[0], 1: y_range[0]}
        },
        {
            'x': {0: x_range[0], 1: x_range[1]},
            'y': {0: y_range[1], 1: y_max_range[1]}
        },
        {
            'x': {0: x_range[1], 1: x_max_range[1]},
            'y': {0: y_range[0], 1: y_range[1]}
        },
    ]

    # plot outside cubes
    for transform_i, transform in enumerate(transforms):
        fig.add_trace(
            go.Mesh3d(
                # 8 vertices of a cube
                x=[transform['x'][coord] for coord in [0, 0, 1, 1, 0, 0, 1, 1]],
                y=[transform['y'][coord] for coord in [0, 1, 1, 0, 0, 1, 1, 0]],
                z=[z_max_range[coord] for coord in [0, 0, 0, 0, 1, 1, 1, 1]],
                opacity=0.2,
                color='rgb(253, 190, 190)',
                i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                hoverinfo='skip',
                name=f'{name}{transform_i}' if name else None
            ),
            row=row, col=col,

        )

    # plot inside cube
    inner_transform = {
        'x': {0: x_range[0], 1: x_range[1]},
        'y': {0: y_range[0], 1: y_range[1]}
    }
    fig.add_trace(
        go.Mesh3d(
            # 8 vertices of a cube
            x=[inner_transform['x'][coord] for coord in [0, 0, 1, 1, 0, 0, 1, 1]],
            y=[inner_transform['y'][coord] for coord in [0, 1, 1, 0, 0, 1, 1, 0]],
            z=[z_max_range[coord] for coord in [0, 0, 0, 0, 1, 1, 1, 1]],
            opacity=0.2,
            color='rgb(236, 74, 74)',
            i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            hoverinfo='skip',
            name=f'{name}{len(transforms)}' if name else None
        ),
        row=row, col=col
    )

def add_benchmark_pt(fig: go.Figure, 
    x: float, 
    y: float, 
    z: float, 
    row:int=1, 
    col:int=1, 
    color:str=None,
    x_range: tuple=(0, 1),
    y_range: tuple=(0, 1),
    z_min: float=0,
    name_pt=None,
    name_lines=None,
):
    """Add point for benchmark along with guiding lines"""
    fig.add_trace(
        go.Scatter3d(
            x=[x], y=[y], z=[z],
            marker=dict(
                color='red' if color is None else color,
                showscale=False
            ),
            name=name_pt
        ),
        row=row, col=col
    )
    add_2dline(fig, x, y, z_min=z_min, color=color, x_range=x_range, y_range=y_range, row=row, col=col, name=name_lines)
    fig.add_trace(
        go.Scatter3d(
            x=[x] * 2,
            y=[y] * 2,
            z=[z, z_min],
            marker=dict(
                size=0,
                color='rgba(0, 0, 0, 0)'
            ),
            line=dict(
                color='red' if color is None else color,
            ),
            hoverinfo='skip',
            name=f'{name_lines}zline' if name_lines else None
        ),
        row=row, col=col
    )

def add_2dline(fig: go.Figure,
    x: float, 
    y: float, 
    z_min: float, 
    color: str=None,
    x_range: tuple=(0, 1),
    y_range: tuple=(0, 1),
    row: int=1, col: int=1,
    name=None
):
    """Add 2 2d lines intersecting at x, y"""
    fig.add_trace(
        go.Scatter3d(
            x=[x] * 2,
            y=y_range,
            z=[z_min] * 2,
            marker=dict(
                size=0,
                color='rgba(0, 0, 0, 0)'
            ),
            line=dict(
                color='red' if color is None else color,
            ),
            hoverinfo='skip',
            name=name
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter3d(
            x=x_range,
            y=[y] * 2,
            z=[z_min] * 2,
            marker=dict(
                size=0,
                color='rgba(0, 0, 0, 0)'
            ),
            line=dict(
                color='red' if color is None else color,
            ),
            hoverinfo='skip',
            name=f'{name}yline' if name else None
        ),
        row=row, col=col
    )
