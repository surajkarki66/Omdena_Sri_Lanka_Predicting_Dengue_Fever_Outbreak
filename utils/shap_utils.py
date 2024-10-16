import shap
import plotly.graph_objects as go
import streamlit.components.v1 as components


from darts import TimeSeries
from darts.explainability.shap_explainer import ShapExplainer


def get_explainer(model: object, background_series: TimeSeries, background_future_covariates: TimeSeries = None, background_num_samples: int = 800) -> ShapExplainer:
    explainer = ShapExplainer(model, background_series=background_series,
                              background_future_covariates=background_future_covariates, background_num_samples=background_num_samples)
    return explainer


def get_shap_explainability(explainer: ShapExplainer, foreground_series: TimeSeries, foreground_future_covariates: TimeSeries = None, horizons: int = 12) -> TimeSeries:
    shap_explainability = explainer.explain(foreground_series=foreground_series,
                                    foreground_future_covariates=foreground_future_covariates, horizons=horizons)
    return shap_explainability

def plot_feature_importance(shap_values: TimeSeries):
    """
    Generate a Plotly line chart for SHAP values over time for multiple features,
    with vertical lines at specified dates.

    Parameters:
    - shap_values: A Darts TimeSeries object containing SHAP values.
    - vertical_lines: A list of timestamps (datetime objects) where vertical lines should be drawn.
    """
    # Extract the SHAP values and feature names
    shap_values_array = shap_values.values()  # This should be (time_steps, features)
    components = shap_values.components.values  # Feature names

    # Check the shape of the SHAP values array
    if shap_values_array.ndim != 2:
        raise ValueError("SHAP values must be a 2D array with shape (time_steps, features).")
    
    # Get the timestamps
    timestamps = shap_values.time_index

    # Create a Plotly figure
    fig = go.Figure()

    # Add a line for each feature
    for i, feature in enumerate(components):
        fig.add_trace(go.Scatter(
            x=timestamps,  # Use timestamps for the x-axis
            y=shap_values_array[:, i].flatten(),  # Flatten to 1D for the current feature
            mode='lines',
            name=feature
        ))

    # # Add vertical lines at specified dates
    for v_line in timestamps:
        fig.add_shape(
            type='line',
            x0=v_line,
            x1=v_line,
            y0=shap_values_array.min(),  # Start at the minimum SHAP value
            y1=shap_values_array.max(),  # End at the maximum SHAP value
            line=dict(color='lightgray', width=1),  # Customize line appearance,
        )

    # Update layout
    fig.update_layout(
        title='SHAP Values Over Forecasted Period for Each Feature',
        xaxis_title='Forecasted Period',
        yaxis_title='SHAP Value',
        xaxis_tickformat='%Y-%m-%d',  # Format the date on the x-axis
        xaxis=dict(
            tickmode='array',
            tickvals=timestamps,  # Show all x-tick values
            ticktext=[str(ts.date()) for ts in timestamps]  # Format tick text
        ),
        showlegend=True,
        legend_title_text="Features"
    )

    # Return the figure object
    return fig


def plot_feature_values(feature_values: TimeSeries):
    """
    Generate a Plotly line chart for feature values over time for multiple features,
    with vertical lines at specified dates.

    Parameters:
    - feature_values: A Darts TimeSeries object containing feature values.
    """
    # Extract the feature values and feature names
    feature_values_array = feature_values.values()  # This should be (time_steps, features)
    components = feature_values.components.values  # Feature names

    # Check the shape of the feature values array
    if feature_values_array.ndim != 2:
        raise ValueError("Feature values must be a 2D array with shape (time_steps, features).")

    # Get the timestamps
    timestamps = feature_values.time_index

    # Create a Plotly figure
    fig = go.Figure()

    # Add a line for each feature
    for i, feature in enumerate(components):
        fig.add_trace(go.Scatter(
            x=timestamps,  # Use timestamps for the x-axis
            y=feature_values_array[:, i].flatten(),  # Flatten to 1D for the current feature
            mode='lines',
            name=feature
        ))

    # Add vertical lines at specified dates, if provided
    for v_line in timestamps:
        fig.add_shape(
            type='line',
            x0=v_line,
            x1=v_line,
            y0=feature_values_array.min(),  # Start at the minimum feature value
            y1=feature_values_array.max(),  # End at the maximum feature value
            line=dict(color='lightgray', width=1),  # Customize line appearance
        )

    # Update layout
    fig.update_layout(
        title='Feature Values Over Forecasted Period for Each Feature',
        xaxis_title='Forecasted Period',
        yaxis_title='Feature Value',
        xaxis_tickformat='%Y-%m-%d',  # Format the date on the x-axis
        xaxis=dict(
            tickmode='array',
            tickvals=timestamps,  # Show all x-tick values
            ticktext=[str(ts.date()) for ts in timestamps]  # Format tick text
        ),
        showlegend=True,
        legend_title_text="Features"
    )

    # Return the figure object
    return fig


def force_plot(shap_values: TimeSeries):
    """
    Generate a Plotly force plot for SHAP values over time for multiple features.

    Parameters:
    - shap_values: A Darts TimeSeries object containing SHAP values.
    """
    # Extract the SHAP values and feature names
    shap_values_array = shap_values.values()  # This should be (time_steps, features)
    components = shap_values.components.values  # Feature names

    # Check the shape of the SHAP values array
    if shap_values_array.ndim != 2:
        raise ValueError("SHAP values must be a 2D array with shape (time_steps, features).")

    # Get the timestamps
    timestamps = shap_values.time_index

    # Create a Plotly figure
    fig = go.Figure()

    # Prepare the data for the force plot
    for i, feature in enumerate(components):
        # Compute the cumulative sum of SHAP values for the current feature
        cumulative_shap = shap_values_array[:, i].cumsum()
        
        # Add a line for the cumulative SHAP values
        fig.add_trace(go.Scatter(
            x=timestamps,  # Use timestamps for the x-axis
            y=cumulative_shap,  # Cumulative SHAP values for the current feature
            mode='lines',
            name=feature
        ))

    # Add a horizontal line at the expected value
    expected_value = shap_values_array.mean(axis=0).cumsum().mean()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[expected_value] * len(timestamps),
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Expected Value'
    ))

    # Update layout
    fig.update_layout(
        title='Cumulative SHAP Values Over Time for Each Feature',
        xaxis_title='Forecasted Period',
        yaxis_title='Cumulative SHAP Value',
        xaxis_tickformat='%Y-%m-%d',  # Format the date on the x-axis
        xaxis=dict(
            tickmode='array',
            tickvals=timestamps,  # Show all x-tick values
            ticktext=[str(ts.date()) for ts in timestamps]  # Format tick text
        ),
        showlegend=True,
        legend_title_text="Features"
    )

    # Return the figure object
    return fig

def st_shap(plot, height=None):
    # Define the style for the body to set text color to white
    style = """
            <style>
                body {
                    color: white;                /* White text */
                }
            </style>
            """
    
    # Construct the HTML with the specified styles and SHAP plot
    shap_html = f"<head>{shap.getjs()}{style}</head><body>{plot.html()}</body>"
    
    # Render the HTML in Streamlit
    components.html(shap_html, height=height)
