"""Shared plotting helpers for the notebooks."""
import plotly.express as px
from plotly.subplots import make_subplots


def stacked_scatter(df, panels, *, x="ts_local", color="name", hover_data=None, height=850, title=None):
    """Scatter panels stacked on one shared time axis, with one legend for all of them.

    panels: [(column, panel title, y-axis title)]. Each sensor keeps one colour in every panel, and clicking a legend
    entry hides it in all panels.
    """
    names = sorted(df[color].dropna().unique())
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=[p[1] for p in panels])
    for row, (col, _, ytitle) in enumerate(panels, start=1):
        sub = px.scatter(df.dropna(subset=[col]), x=x, y=col, color=color, hover_data=hover_data,
                         category_orders={color: names})
        for tr in sub.data:
            tr.update(legendgroup=tr.name, showlegend=(row == 1))
            fig.add_trace(tr, row=row, col=1)
        fig.update_yaxes(title_text=ytitle, row=row, col=1)
    fig.update_layout(height=height, title=title, legend=dict(font=dict(size=10), itemsizing="constant", title_text=""),
                      margin=dict(r=10))
    return fig
